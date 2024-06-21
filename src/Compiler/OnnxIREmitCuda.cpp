//===- OnnxIREmitCuda.cpp - Translating to Cuda calls ----------------------===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Compiler/OnnxIREmitCuda.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"


#include <utility>
#include <stack>
#include <set>
#include <map>
#include "../Dialect/Mlir/DialectBuilder.hpp"
#include <llvm/IR/DerivedTypes.h>



#define DEBUG_TYPE "translate-to-cuda"

using namespace mlir;
using llvm::formatv;


#define ENABLE_INFO_QT 0

#if ENABLE_INFO_QT
#define INFO(info) os << "//INFO: " << info << "\n";
#else
#define INFO(info)
#endif /* ENABLE_INFO_QT */

/// Options
bool enableStreamAndEvent = true;
bool useCustomPPL = false;
bool enableTiming = true;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace onnx_mlir {

/// Emitter that uses dialect specific emitters to emit Cuda code.
struct CudaEmitter {
  explicit CudaEmitter(raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);


  /// Emit ssa var declaration or return falure.
  LogicalResult emitDeclaration(Value &value);

  /// Emit ssa tensor var device malloc or return falure.
  LogicalResult emitDeviceMalloc(Value &value);

  /// Emit ssa var free or return falure.
  LogicalResult emitFree(Value &value);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  /// Emits a declaration of a variable with the given type and name.
  LogicalResult emitVariableDeclaration(Location loc, Type type,
                                        StringRef name);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Emits value as an operands of an operation
  LogicalResult emitOperand(Value value);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CudaEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper),
          lifetimeTrackerFirstScope(emitter.valueFisrtUseTracker),
          lifetimeTrackerLastScope(emitter.valueLastUseTracker),
          emitter(emitter)
        {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, unsigned int> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    llvm::ScopedHashTableScope<Value, Operation *> lifetimeTrackerFirstScope;
    llvm::ScopedHashTableScope<Value, Operation *> lifetimeTrackerLastScope;
    CudaEmitter &emitter;
  };

  /// Return the existing or a new name for a Value.
  std::string getOrCreateName(Value val);
  unsigned int getOrCreateIndex(Value val);

  /// Returns wether the Value is assigned to a variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

  /// Emit include
  void emitInclude(StringRef name, bool isLocal);

  /// Collect and print ppl.cv related include
  void collectPplInc(CudaEmitter &emitter, ModuleOp moduleOp);
  void printPplInc(CudaEmitter &emitter);

  /// Insert and check for ppl op type
  void insertPplOp(StringRef name) {
    onnx_op_types.insert(name);
  };
  bool hasPplOp(StringRef name) {
    return (onnx_op_types.find(name) != onnx_op_types.end());
  };

  /// Track lifetime of ssa
  LogicalResult trackValueLifetime(func::FuncOp funcOp);
  void updateValueUse(Value value, Operation *op) {
    //#### 最后调用
    //1. `map(key=Value, val=Operation)`
    //2. `funcOp.walk`
    //3. `map[Value]=Operation`
    //4. `map`里是SSA的最后调用
    //#### 最初调用
    //1. `map(key=Value, val=Operation)`
    //2. `funcOp.walk`
    //3. `if (!map[Value].exist) { map[Value] = Operation; }`
    //4. `map`里是SSA的最初调用
    if(!valueFisrtUseTracker.count(value)) {
      valueFisrtUseTracker.insert(value, op);
    }
    valueLastUseTracker.insert(value, op);
  };
  Operation *getValueFirstUse(Value value);
  Operation *getValueLastUse (Value value);

  //Pre/Post proccess for every ONNX op. 
  LogicalResult emitONNXPreOp (Operation &op);
  LogicalResult emitONNXPostOp(Operation &op);

  std::string getStreamName(Value value);
  std::string getEventName(Value value);
  std::string getConstantName(Value value);

  //Push/Pop cudaEvent record.
  void insertCudaEvent(Value value, int count) {
    eventRecord[getOrCreateName(value)] = count;
  };
  int  numRefCudaEvent(Value value) {
    if ( eventRecord.find(getOrCreateName(value)) != eventRecord.end()) {
      return eventRecord[getOrCreateName(value)];
    }
    return 0;
  };
  void dropCudaEvent(Value value) {
    Value v0 = value.getDefiningOp()->getResult(0);
    if(numRefCudaEvent(v0)) {
      (*eventRecord.find(getOrCreateName(v0))).second--;
    }
    else { return; }
    if(0 == eventRecord[getOrCreateName(v0)]) { eventRecord.erase(getOrCreateName(v0)); }
  };
  std::string popCudaEvent() {
    if (!eventRecord.empty()) {
      std::string res = (*eventRecord.begin()).first;
      eventRecord.erase(res);
      return res;
    } else {
      return NULL;
    }
  }
  bool hasCudaEvent(void) {
    return !eventRecord.empty();
  };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, unsigned int>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;
  using LifeTimeTracker = llvm::ScopedHashTable<Value, Operation *>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  /// Record all events
  std::map<std::string, int> eventRecord;

  /// Collect all onnx op using ppl.cv to print include lines
  /// eg. mlir::ONNXAbsOp -> #include <ppl/cv/cuda/Abs.h>
  std::set<StringRef> onnx_op_types;

  /// Track first and last use of Value
  LifeTimeTracker valueFisrtUseTracker;
  LifeTimeTracker valueLastUseTracker;

public:
  LogicalResult emitPPLShapeDeclaration(Value &value);
  std::string getPPLShapeName(Value &value) {
    return getOrCreateName(value) + "Shape";
  };
  LogicalResult emitPPLType(Type type);

private:
  std::string pplCommonPrefix = "ppl::common::";

};

CudaEmitter::CudaEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

unsigned int CudaEmitter::getOrCreateIndex(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val,++valueInScopeCount.top());
  return *valueMapper.begin(val);
}

/// Return the existing or a new name for a Value.
std::string CudaEmitter::getOrCreateName(Value val) {
  return formatv("v{0}", getOrCreateIndex(val));
}

LogicalResult CudaEmitter::emitType(::mlir::Location loc, ::mlir::Type type) {
  if (auto tType = type.dyn_cast<TensorType>()) {
    os << "/*" << type << "*/ ";
    if(failed(emitType(loc, tType.getElementType()))) {
      return emitError(loc, "cannot emit tensor element type ") << type;
    }
    return (os << " *"), success();
  }
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 16: 
      return (os << "half"), success();
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto nType = dyn_cast<NoneType>(type)) {
    os << "[[maybe_unused]] int ";
    return success();
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  // FIXME: emit every type for now. //return emitError(loc, "cannot emit unkown type ") << type;
  os << type;
  return success();
}

bool CudaEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool CudaEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CudaEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult CudaEmitter::emitAttribute(Location loc, Attribute attr) {
  //qt[24/4/28]: ignore attr for now.
  return success();
}

LogicalResult CudaEmitter::trackValueLifetime(func::FuncOp funcOp) {
  funcOp.walk([&](Operation *op) {
    for (auto i : op->getOperands()) {
      updateValueUse(i, op);
    }
  });
  return success();
}

Operation *CudaEmitter::getValueFirstUse(Value value) {
  Value v = value.getDefiningOp() ? value.getDefiningOp()->getResult(0) : value; 
  if (!valueFisrtUseTracker.count(v)) {
    return NULL;
  }
  return valueFisrtUseTracker.lookup(v);
}

Operation *CudaEmitter::getValueLastUse (Value value) {
  if (!valueLastUseTracker.count(value)) {
    return NULL;
  }
  return valueLastUseTracker.lookup(value);
}

LogicalResult CudaEmitter::emitPPLType(Type type) {
  //enum {
  //    DATATYPE_UNKNOWN = 0,
  //    DATATYPE_UINT8 = 1,
  //    DATATYPE_UINT16 = 2,
  //    DATATYPE_UINT32 = 3,
  //    DATATYPE_UINT64 = 4,
  //    DATATYPE_FLOAT16 = 5,
  //    DATATYPE_FLOAT32 = 6,
  //    DATATYPE_FLOAT64 = 7,
  //    DATATYPE_BFLOAT16 = 8,
  //    DATATYPE_INT4B = 9,
  //    DATATYPE_INT8 = 10,
  //    DATATYPE_INT16 = 11,
  //    DATATYPE_INT32 = 12,
  //    DATATYPE_INT64 = 13,
  //    DATATYPE_BOOL = 14,
  //    DATATYPE_COMPLEX64 = 15,
  //    DATATYPE_COMPLEX128 = 16,
  //    DATATYPE_MAX,
  //};
  //typedef uint32_t datatype_t;
  os << pplCommonPrefix;
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "DATATYPE_BOOL"), success();
    case 8:  // Shared
    case 16: // Shared
    case 32: // Shared
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "DATATYPE_UINT" << iType.getWidth()), success();
      else
        return (os << "DATATYPE_INT" << iType.getWidth()), success();
    default:
      return failure();
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 16: // Shared
    case 32: // Shared
    case 64:
      return (os << "DATATYPE_FLOAT" << fType.getWidth()), success();
    default:
      return failure();
    }
  }

  return failure();
}

LogicalResult CudaEmitter::emitPPLShapeDeclaration(Value &value) {
  //ppl::common::TensorShape shape0, shape1, shape2;
  //shape0.SetDimCount(3);
  //shape0.SetDim(0, 5);
  //shape0.SetDim(1, 6);
  //shape0.SetDim(2, 3);
  //shape0.SetDataType(ppl::common::DATATYPE_FLOAT32);
  TensorType tType = dyn_cast_if_present<TensorType>(value.getType());
  if (!tType) {
    return emitError(value.getLoc(), "value is not TensorType!");
  }

  if (tType.getNumDynamicDims()) {
    return (os << "ERROR: Unsupported Dynamic Tensor!\n"), success();
    return emitError(value.getLoc(), "cannot print dynamic tensor shape for ppl!");
  }

  //Assumption: shape must be static(inferenced).
  std::string shapeName =  getPPLShapeName(value);
  os << "ppl::common::TensorShape " << shapeName << ";\n";
  os << shapeName << ".SetDimCount(" << tType.getRank() << ");\n";
  for (auto i = 0; i < tType.getRank(); i++) {
    os << shapeName << ".SetDim(" << i << ", " << tType.getDimSize(i) << ");\n";
  }
  os << shapeName << ".SetDataType(";
  if (failed(emitPPLType(tType.getElementType()))) {
    return emitError(value.getLoc(), "fail to emit pplType!");
  }
  os << ");\n";

  return success();
}


void printShape(CudaEmitter &emitter, TensorType tType) {
  raw_indented_ostream &os = emitter.ostream();
  os << "{";
  for (auto i = 0; i < tType.getRank(); i++) {
    os << ( (i == 0)? "" : ", ") << (tType.isDynamicDim(i) ? "Dyn" : std::to_string(tType.getDimSize(i)));
  }
  os << "}";
}

static LogicalResult printCallOperation(CudaEmitter &emitter, Operation *callOp,
                                        StringRef callee) {

  raw_indented_ostream &os = emitter.ostream();
  os << callee << "(";
  os << "TODO: args";
  os << ")";
  return success();
}

/// print fixed code: include and leakyrelu implement
void printFixedCode(CudaEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  os << "#include <cuda_runtime.h>\n";
  os << "#include <cstdlib>\n";
  os << "#include <iostream>\n";
  os << "#include <vector>\n";
  return;
}

LogicalResult printOperation(CudaEmitter &emitter,::mlir::ONNXAbsOp absOp) {
  //TODO:
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXLeakyReluOp leakyReluOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value x = leakyReluOp.getX();
  Value y = leakyReluOp.getY();
  llvm::APFloat alpha = leakyReluOp.getAlpha();
  auto tTypeX = x.getType().dyn_cast_or_null<TensorType>();
  auto tTypeY = y.getType().dyn_cast_or_null<TensorType>();

  if ((!tTypeX) || (!tTypeY)) {
    return leakyReluOp.emitOpError("operand not valid tensor type!");
  }

  os << "onnxLeakyReLU";
  os << "<<<(";
  tTypeX.getNumElements(); 
  os << " + threads_per_block - 1)/threads_per_block, threads_per_block>>>";
  os << "(";
  os << emitter.getOrCreateName(x); //X
  os << ", ";
  os << emitter.getOrCreateName(y); //Y
  os << ", ";
  tTypeX.getNumElements();
  os << ", ";
  os << (tTypeY.getElementTypeBitWidth() >> 3); //stride
  os << ", ";
  os << alpha.convertToFloat(); //alpha
  os << ");";
  return success();
}

template <typename T>
LogicalResult printONNXArithmeticPPLCudaKernel(CudaEmitter &emitter, T arithmeticOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value a = arithmeticOp.getA();
  Value b = arithmeticOp.getB();
  Value c = arithmeticOp.getC();
  auto tTypeA = a.getType().dyn_cast_or_null<TensorType>();
  auto tTypeB = b.getType().dyn_cast_or_null<TensorType>();
  auto tTypeC = c.getType().dyn_cast_or_null<TensorType>();
  auto stream = emitter.getStreamName(c);

  //   PPLCUDAArithMeticAddForwardImp(stream,
  //     &shapeA, a,
  //     &shapeB, b,
  //     &shapeC, c);
  os << "PPLCUDAArithMetic";
  if      (dyn_cast_or_null<mlir::ONNXAddOp>(&arithmeticOp)) { os << "Add";      }
  else if (dyn_cast_or_null<mlir::ONNXSubOp>(&arithmeticOp)) { os << "Sub";      }
  else if (dyn_cast_or_null<mlir::ONNXMulOp>(&arithmeticOp)) { os << "Mul";      }
  else if (dyn_cast_or_null<mlir::ONNXDivOp>(&arithmeticOp)) { os << "Div";      }
  else { return arithmeticOp.emitError("op is not onnx arithmetic!");            }
  os << "ForwardImp(";
  os << stream                              << ", "; // stream
  os << "&" << emitter.getPPLShapeName(a)   << ", "; // shape of A
  os << emitter.getOrCreateName(a)          << ", "; // A
  os << "&" << emitter.getPPLShapeName(b)   << ", "; // shape of B
  os << emitter.getOrCreateName(b)          << ", "; // B
  os << "&" << emitter.getPPLShapeName(c)   << ", "; // shape of C
  os << emitter.getOrCreateName(c)                 ; // C
  //os << 1.0f                                << ", "; // scale A
  //os << 1.0f                                << ", "; // scale B
  //os << 1.0f                                ; // scale C
  os << ");";

  os.indent();
  os << "\n";
  os << "//A "; printShape(emitter, tTypeA); os << ";\n";
  os << "//B "; printShape(emitter, tTypeB); os << ";\n";
  os << "//C "; printShape(emitter, tTypeC); os << ";\n";
  os.unindent();

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXAddOp addOp) {
  return printONNXArithmeticPPLCudaKernel<ONNXAddOp>(emitter, addOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXMulOp mulOp) {
  return printONNXArithmeticPPLCudaKernel<ONNXMulOp>(emitter, mulOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXSubOp subOp) {
  return printONNXArithmeticPPLCudaKernel<ONNXSubOp>(emitter, subOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXDivOp divOp) {
  return printONNXArithmeticPPLCudaKernel<ONNXDivOp>(emitter, divOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXPowOp powOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value a = powOp.getX();
  Value b = powOp.getY();
  Value c = powOp.getZ();
  auto tTypeA = a.getType().dyn_cast_or_null<TensorType>();
  auto tTypeB = b.getType().dyn_cast_or_null<TensorType>();
  auto tTypeC = c.getType().dyn_cast_or_null<TensorType>();
  auto stream = emitter.getStreamName(c);

  //   PPLCUDAArithMeticAddForwardImp(stream,
  //     &shapeA, a,
  //     &shapeB, b,
  //     &shapeC, c);
  os << "PPLCUDAArithMetic";
  os << "Pow";
  os << "ForwardImp(";
  os << stream                              << ", "; // stream
  os << "&" << emitter.getPPLShapeName(a)   << ", "; // shape of A
  os << emitter.getOrCreateName(a)          << ", "; // A
  os << "&" << emitter.getPPLShapeName(b)   << ", "; // shape of B
  os << emitter.getOrCreateName(b)          << ", "; // B
  os << "&" << emitter.getPPLShapeName(c)   << ", "; // shape of C
  os << emitter.getOrCreateName(c)                 ; // C
  //os << 1.0f                              << ", "; // scale A
  //os << 1.0f                              << ", "; // scale B
  //os << 1.0f                                     ; // scale C
  os << ");";

  os.indent();
  os << "\n";
  os << "//A "; printShape(emitter, tTypeA); os << ";\n";
  os << "//B "; printShape(emitter, tTypeB); os << ";\n";
  os << "//C "; printShape(emitter, tTypeC); os << ";\n"; 
  os.unindent();

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXMaxPoolSingleOutOp maxPoolSingleOutOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = maxPoolSingleOutOp.getResult();
  Value inp = maxPoolSingleOutOp.getOperand();

  size_t dims = useCustomPPL ? 3 : 2;
  int kernel[dims];
  int stride[dims];
  int pad[dims];

  for (size_t i = 0; i < dims; i++ ) {
    kernel[i] = maxPoolSingleOutOp.getKernelShape().size() > i ? maxPoolSingleOutOp.getKernelShape()[i].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 1;
    stride[i] = maxPoolSingleOutOp.getStrides().has_value() && maxPoolSingleOutOp.getStrides().value().size() > i ?
      maxPoolSingleOutOp.getStrides().value()[i].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 1;
    pad[i]    = maxPoolSingleOutOp.getPads().has_value()  &&  maxPoolSingleOutOp.getStrides().value().size() > i ?
      maxPoolSingleOutOp.getPads().value()[i].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 0;
  }

  os << "PPLCUDAMaxPoolingForwardImp(";  //ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
  os << emitter.getStreamName(res) << ", ";          //  cudaStream_t stream,
  os << "&" << emitter.getPPLShapeName(inp) << ", "; //  ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(inp) << ", ";        //  const void* input,
  os << "&" << emitter.getPPLShapeName(res) << ", "; //  ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(res) << ", ";        //  void* output,
  os << kernel[0] << ", ";                           //  int kernel,
  os << kernel[1] << ", ";                           //  int kernel,
  if (useCustomPPL) 
    os << kernel[2] << ", ";                         //  int kernel,
  os << stride[0] << ", ";                           //  int stride,
  os << stride[1] << ", ";                           //  int stride,
  if (useCustomPPL) 
    os << stride[2] << ", ";                         //  int stride,
  os << pad[0] << ", ";                              //  int padding,
  os << pad[1] << ", ";                              //  int padding,
  if (useCustomPPL) 
    os << pad[2] << ", ";                            //  int padding,

  os << 1.0f << ", ";                                //  float in_scale,
  os << 1.0f << ");\n";                              //  float out_scale);

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXConcatOp concatOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = concatOp.getResult();
  auto operands = concatOp.getOperands();
 
  INFO("collect input_dims and input_padded_dims, since ONNX.concat does not have pad attr, these 2 same.")
  INFO("need to convert int64 * dims to int32 *")

  std::string pplInputDimsName = emitter.getOrCreateName(res) + "pplInputDims";

  // collect shapes
  os << "pplInputShapes.clear();\n";
  for (auto i : operands) {
    os << "pplInputShapes.push_back(" << emitter.getPPLShapeName(i) << ");\n";
  }

  os << "int **" << pplInputDimsName << " = createPPLDims(pplInputShapes);\n";

  INFO("collect inputs.");
  os << "const void *" << emitter.getOrCreateName(res) << "ONNXConcatOpInputs[] = {";
  for (auto i : operands) {
    os << emitter.getOrCreateName(i) << ", ";
  }
  os << "};\n";

  std::string stream = emitter.getStreamName(res);
  auto axis = concatOp.getAxis();
  auto numInput = concatOp.getNumOperands();

  os << "PPLCUDAConcatForwardImp(";                                         //  ppl::common::RetCode PPLCUDAConcatForwardImp(
  os << stream                                                    << ", ";  //      cudaStream_t stream,
  os << axis                                                      << ", ";  //      int axis,
  os << numInput                                                  << ", ";  //      int num_inputs,
  os << pplInputDimsName                                          << ", ";  //      int* input_dims[],
  os << pplInputDimsName                                          << ", ";  //      int* input_padded_dims[],
  os << emitter.getOrCreateName(res) << "ONNXConcatOpInputs"      << ", ";  //      const void* inputs[],
  os <<  "&" << emitter.getPPLShapeName(res)                      << ", ";  //      ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(res)                              << ", ";  //      void* output,
  os << "0"                                                       << ");";  //      int mask = 0);
  os << "\n";

  os << "destroyPPLDims(" << pplInputDimsName;
  os << ", pplInputShapes.size());\n";
  os << "pplInputShapes.clear();\n";

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXConstantOp constantOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = constantOp.getResult();
  size_t size = 0;
  Type type = res.getType();
  std::string constName = emitter.getConstantName(res);

  if (constantOp.getSparseValue().has_value()) {
    return constantOp.emitError("Only support dense values at this time");
  }
  assert(constantOp.getValue().has_value() && "Value is not set");

  if (auto tType = dyn_cast_if_present<TensorType>(type)) {
    os << "extern const ";
    if (failed(emitter.emitType(res.getLoc(), tType.getElementType()))) {
      return failure();
    }
    os << " " << emitter.getConstantName(res) << "[];\n";
    size = tType.getNumElements() * tType.getElementTypeBitWidth() / 8;
  }

#if 0 /* keep for simple constant type handling */
  bool showConstContent = true;
  if (auto tensorAttr = constantOp.getValueAttr().dyn_cast<DisposableElementsAttr>()) {
    if (auto tType = res.getType().dyn_cast<TensorType>()) {
      auto eType = tType.getElementType();
      os << "extern const ";
      if (failed(emitter.emitType(res.getLoc(), eType))) {
        return failure();
      }
      os << " " << constName << "[];{"; // FIXME: os << " = {" 
      if (1) { size = 16; } // FIXME:" 
      else if (eType.isa<Float16Type>()) {
        auto t =  tensorAttr.getArray<float_16>();
        auto t1 = t.get();
        for(auto i : t1) { os << i.toFloat() << ", "; if(!showConstContent) { break; }}
        size = t1.size() * sizeof(float_16);
      } else if (eType.isa<Float32Type>()) {
        auto t =  tensorAttr.getArray<float>();
        auto t1 = t.get();
        for(auto i : t1) { os << i << ", "; if(!showConstContent) { break; }}
        size = t1.size() * sizeof(float);
      } else if (eType.isa<IntegerType>()) {
        auto iType = eType.dyn_cast<IntegerType>();
        if (iType.getWidth() == 64) {
          auto t =  tensorAttr.getArray<int64_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; if(!showConstContent) { break; }}
          size = t1.size() * sizeof(int64_t);
        } else if (iType.getWidth() == 32) {
          auto t =  tensorAttr.getArray<int32_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; if(!showConstContent) { break; }}
          size = t1.size() * sizeof(int);
        } else if (iType.getWidth() == 16) {
          auto t =  tensorAttr.getArray<int16_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; if(!showConstContent) { break; }}
          size = t1.size() * sizeof(int16_t);
        } else if (iType.getWidth() == 8) {
          auto t =  tensorAttr.getArray<int8_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << (int)i << ", "; if(!showConstContent) { break; }}
          size = t1.size() * sizeof(int8_t);
        } else {
          os << "WTF: ??? " << res.getLoc() << "\n"; 
        }
      } else {
        os << "WTF: ??? " << res.getLoc() << "\n";
      }
      
      os << "};\n";
    }
  } else if (auto tensorAttr = constantOp.getValueAttr().dyn_cast<DenseElementsAttr>()) {
    if (TensorType tType = type.dyn_cast<TensorType>()) {
      Type eType = tType.getElementType();
      if (auto intDenseAttr = tensorAttr.dyn_cast_or_null<DenseIntElementsAttr>()) {
        if (auto iType = eType.dyn_cast<IntegerType>()) {
          auto valueRange = intDenseAttr.getValues<IntegerAttr>();
          os << "static const ";

          if (failed(emitter.emitType(res.getLoc(), iType))) {
            return constantOp.emitError("emit dense ints const value failed!");
          }
          os << " " << constName << "[] = {";
          for (auto i : valueRange) {
            os << *i.getValue().getRawData() << ", ";if(!showConstContent) { break; }
          }
          os <<"};\n";
          size = valueRange.size() * iType.getWidth() / 8;
        }
      } else if (auto floatDenseAttr = tensorAttr.dyn_cast_or_null<DenseFPElementsAttr>()) {
        auto valueRange = floatDenseAttr.getValues<FloatAttr>();
        if (auto fType = eType.dyn_cast<FloatType>()) {
          os << "static const ";
          if (failed(emitter.emitType(res.getLoc(), fType))) {
            return constantOp.emitError("emit dense ints const value failed!");
          }
          os << " " << constName << "[] = {";
          for (auto i : valueRange) {
            os << i.getValue().convertToFloat() << ", ";if(!showConstContent) { break; }
          }
          os <<"};\n";
          size = valueRange.size() * fType.getWidth() / 8;
        }
      }
    }
  } else if (constantOp.getValueFloat().has_value()) {
    float f = constantOp.getValueFloat().value().convertToFloat();
    os << emitter.getOrCreateName(res) << " = " << f << ";\n";
  } else if (constantOp.getValueFloats().has_value()) {
    auto fs = constantOp.getValueFloats().value();
    os << "static const float " << constName << "[] = {";
    for (auto i : fs.getValue()) {
      os << i.cast<FloatAttr>().getValue().convertToFloat() << ", ";if(!showConstContent) { break; }
    }
    os << "};\n";
    size = sizeof(float) * fs.size();
  } else if (constantOp.getValueInt().has_value()) {
    int i = constantOp.getValueInt().value();
    os << emitter.getOrCreateName(res) << " = " << i << ";\n";
  } else if (constantOp.getValueInts().has_value()) {
    auto fs = constantOp.getValueInts().value().getValue();
    os << "static const int " << constName << "[] = {";
    for (auto i : fs) {
      os << i.dyn_cast<IntegerAttr>().getValue() << ", ";if(!showConstContent) { break; }
    }
    os << "};\n";
    size = sizeof(int) * fs.size();
  } else if (constantOp.getValueString().has_value()) {
    os << "char *" << constName << "[] = \"";
    os <<  constantOp.getValueString().value() << "\"\n";
    size = constantOp.getValueString().value().size();
  } else if (constantOp.getValueStrings().has_value()) {
    return constantOp.emitError("wtf wtf wtf!");
  } else {
    llvm::errs() << "constant type : " << res.getType() << "\n";
    //llvm::errs() << "type : " << constantOp.getValueAttr() << "\n";
    return constantOp.emitError("string list and other constant type not supported yet!");
  }
#endif /* 0 */

  if (size) {
    if (enableStreamAndEvent) {
      os << "cudaMemcpyAsync(";
    } else {
      os << "cudaMemcpy(";
    }
    os << emitter.getOrCreateName(res) << ", "; //dst
    os << constName << ", "; //src
    os << size << ", ";
    os << "cudaMemcpyHostToDevice, ";
    if (enableStreamAndEvent) {
      os << emitter.getStreamName(res);
    }
    os << ");\n";
  }

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, mlir::ONNXReshapeOp reshapeOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value input = reshapeOp.getOperand(0);
  Value shape = reshapeOp.getOperand(1);
  Value res   = reshapeOp.getResult();
  TensorType tType = shape.getType().dyn_cast<TensorType>();
  if (!tType) {
    return reshapeOp.emitError("target shape is not tensor!");
  }

  std::string shapeHostName = emitter.getOrCreateName(shape) + "HostCopyFor" + emitter.getOrCreateName(res);
  if (failed(emitter.emitType(shape.getLoc(), shape.getType()))) {
    return failure();
  }
  os << shapeHostName << "=(";
  if (failed(emitter.emitType(shape.getLoc(), shape.getType()))) {
    return failure();
  }
  os << ")malloc(" << tType.getNumElements();
  os << " * sizeof(";
  if (failed(emitter.emitType(shape.getLoc(), tType.getElementType()))) {
    return failure();
  }
  os << "));\n";

  os << "cudaMemcpyAsync(";
  os << shapeHostName << ", ";
  os << emitter.getOrCreateName(shape) << ", ";
  os << tType.getNumElements();
  os << " * sizeof(";
  if (failed(emitter.emitType(shape.getLoc(), tType.getElementType()))) {
    return failure();
  }
  os << "), ";
  os << "cudaMemcpyDeviceToHost, ";
  os << emitter.getStreamName(res);
  os << ");\n";


  std::string shapeName =  emitter.getOrCreateName(shape) + "ValueToShapeFor" + emitter.getOrCreateName(res);
  os << "ppl::common::TensorShape " << shapeName << ";\n";
  os << shapeName << ".SetDimCount(" << tType.getDimSize(0) << ");\n";
  for (auto i = 0; i < tType.getDimSize(0); i++) {
    os << shapeName << ".SetDim(" << i << ", " << shapeHostName << "[" << i << "]" << ");\n";
  }
  os << shapeName << ".SetDataType(";
  if (failed(emitter.emitPPLType(tType.getElementType()))) {
    return emitError(shape.getLoc(), "fail to emit pplType!");
  }
  os << ");\n";

  os << "PPLCUDAReshapeForwardImp(";                    //ppl::common::RetCode PPLCUDAReshapeForwardImp(
  os << emitter.getStreamName(res) << ", ";             //  cudaStream_t stream,
  os << "&" << emitter.getPPLShapeName(input) << ", ";  //  const ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(input) << ", ";         //  const void* input,
  os << "&" << shapeName << ", ";                       //  const ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(res) << ");\n";         //  void* output);

  os << "free(" << shapeHostName << ");\n";
  return success();
}

int pplResizeTranformMode(StringRef mode) {
  if (mode == "half_pixel") {
    return 0;
  } else if (mode == "half_pixel_symmetric") {
    return -1; //not supported
  } else if (mode == "pytorch_half_pixel") {
    return 1;
  } else if (mode == "align_corners") {
    return 2;
  } else if (mode == "asymmetric") {
    return 3;
  } else if (mode == "tf_crop_and_resize") {
    return 5;
  }
  return -1;
}

int pplResizeInterMode(StringRef mode) {
  if (mode == "nearest") {
    return 0;
  } else if (mode == "linear") {
    return 1;
  } else if (mode == "cubic") {
    return 2;
  }
  return -1;
}

int pplResizeNearestMode(StringRef mode) {
  //deps/ppl.kernel.cuda/src/nn/resize.cu::PPLCUDAResizeForwardImp : nearest_mode not used...
  if (mode == "round_prefer_floor") {
    
  } else if (mode == "round_prefer_ceil") {

  }  else if (mode == "floor") {

  }  else if (mode == "ceil") {

  }
  return -1;
}


/// only support scales, roi&sizes not suppoerted yet.
LogicalResult printOperation(CudaEmitter &emitter, mlir::ONNXResizeV13Op resizeOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = resizeOp.getResult();
  Value input = resizeOp.getX();
  Value scales = resizeOp.getScales();
  Value roi = resizeOp.getRoi();
  int transformMode = pplResizeTranformMode(resizeOp.getCoordinateTransformationMode());
  int interMode = pplResizeInterMode(resizeOp.getMode());
  int nearestMode = pplResizeNearestMode(resizeOp.getNearestMode());

  if (resizeOp.getScales().getDefiningOp()->getName().getStringRef() != "onnx.Constant") {
    return resizeOp.emitError("dynamic resize scale not supported!");
  }
  if (transformMode < 0) {
    return resizeOp.emitError("transform mode not supported!");
  }
  if (interMode < 0) {
    return resizeOp.emitError("invalid inter mode!");
  }

  os << "PPLCUDAResizeForwardImp(";          //ppl::common::RetCode PPLCUDAResizeForwardImp(
  os << emitter.getStreamName(res) << ", ";                       //  cudaStream_t stream,
  os <<  "&" << emitter.getPPLShapeName(input) << ", ";           //  const ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(input) << ", ";                   //  const void* input,
  os <<  "&" << emitter.getPPLShapeName(res) << ", ";             //  const ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(input) << ", ";                   //  void* outData,
  os << "false" << ", ";                                          //  bool scale_pre_set,
  os << emitter.getConstantName(scales) << "[2], ";               //  float h_scale,
  os << emitter.getConstantName(scales) << "[3], ";               //  float w_scale,
  if (useCustomPPL) {
    if (roi.getDefiningOp() && isa<ONNXConstantOp>(roi.getDefiningOp())) {
      os << emitter.getConstantName(roi) << "[0], ";              // float roi
      os << emitter.getConstantName(roi) << "[1], ";              // float roi
      os << emitter.getConstantName(roi) << "[2], ";              // float roi
      os << emitter.getConstantName(roi) << "[3], ";              // float roi
    } else {
      os << "0.f, 0.f, 1.f, 1.f, ";
    }
    os << (int)resizeOp.getExcludeOutside() << ", ";
  }
  os << transformMode << ", ";                                    //  int transform_mode,
  os << interMode << ", ";                                        //  int inter_mode,
  os << resizeOp.getCubicCoeffA().convertToFloat() << ", ";       //  float cubic_coeff,
  os << nearestMode << ", ";                                      //  int nearest_mode,
  os << 1.f << ", ";                                              //  float in_scale,
  os << 1.f;                                                      //  float out_scale,
  if (useCustomPPL)                                             
    os << ", " << resizeOp.getExtrapolationValue().convertToFloat();      //  ExtrapolationValue
  os << ");\n";
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, mlir::ONNXSigmoidOp sigmoidOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = sigmoidOp.getY();
  Value input = sigmoidOp.getX();

  os << "PPLCUDAUnarySigmoidForwardImp("; //ppl::common::RetCode PPLCUDAUnarySigmoidForwardImp(
  os << emitter.getStreamName(res)                    << ", "; //  cudaStream_t stream,
  os << "&" << emitter.getPPLShapeName(input)         << ", "; //  const ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(input)                << ", "; //  const void* input,
  os << "&" << emitter.getPPLShapeName(res)           << ", "; //  const ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(input)                       ; //  void* output,
  os << ");\n";                                                //  const QuantKernelParamCuda* qparam = nullptr);

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter,  mlir::ONNXSplitV13Op splitOp) {
  raw_indented_ostream &os = emitter.ostream();
  std::vector<Value> results;
  Value input = splitOp.getInput();

  os << "pplInputShapes.clear();\n";
  std::string pplSplitOutDimsName = emitter.getOrCreateName(splitOp.getResults()[0]) + "pplSplitOutDims";


  for (auto i : splitOp.getResults()) {
    results.push_back(i);
    os << "pplInputShapes.push_back(" << emitter.getPPLShapeName(i) << ");\n";
  }

  os << "const int64_t **" <<pplSplitOutDimsName << "= createPPLDimsI64(pplInputShapes);\n";
  os << "void *" << emitter.getOrCreateName(results[0]) << "SplitOutputs[] = {";
  for (auto i : results) {
    os << emitter.getOrCreateName(i) << ", ";
  }
  os << "};\n";
  
  os << "PPLCUDASplitForwardImp(";                                        //ppl::common::RetCode PPLCUDASplitForwardImp(
  os << emitter.getStreamName(results[0]) << ", ";                        //  cudaStream_t stream,
  os << (int)splitOp.getAxis() << ", ";                                        //  int split_axis,
  os << "&" << emitter.getPPLShapeName(input) << ", ";                           //  const ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(input) << ", ";                           //  const void* input,
  os << splitOp.getNumResults() << ", ";                                  //  int num_outputs,
  os << pplSplitOutDimsName << ", ";                                           //  const int64_t* out_dims[],
  os << emitter.getOrCreateName(results[0]) << "SplitOutputs" << ");\n";  //  void* output[]);

  os << "destroyPPLDimsI64(" << pplSplitOutDimsName << ", pplInputShapes.size());\n";
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, mlir::ONNXTransposeOp transposeOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value Y = transposeOp.getResult();
  Value X = transposeOp.getOperand();
  auto perm = transposeOp.getPerm();

  std::string kernelParamName = emitter.getOrCreateName(Y) + "transposeKernelParam";
  os << "TransposeKernelParam " << kernelParamName << ";\n";
  if (perm.has_value()) {
    for (auto i : perm.value()) {
      os << kernelParamName << ".perm.push_back(" << i.dyn_cast<IntegerAttr>().getValue() << ");\n";
    }
  } else {
    for (auto i = 0; i < (int)perm.value().size(); i++) {
      os << kernelParamName << ".perm.push_back(" << i << ");\n";
    }
  }

  os << "PPLCUDATransposeForwardImp(";              //ppl::common::RetCode PPLCUDATransposeForwardImp(
  os << emitter.getStreamName(Y) << ", ";           //    cudaStream_t stream,
  os << kernelParamName << ", ";                    //    TransposeKernelParam param,
  os << "&" << emitter.getPPLShapeName(X) << ", ";  //    const ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(X) << ", ";         //    const void* input,
  os << "&" << emitter.getPPLShapeName(Y) << ", ";  //    const ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(Y) << ");\n";       //    void* output);

  return success();
}

LogicalResult printCustomConv(CudaEmitter &emitter, mlir::ONNXConvOp convOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value input = convOp.getOperand(0);
  Value weight= convOp.getOperand(1);
  Value bias = convOp.getNumOperands() > 2 ? convOp.getOperand(2) : NULL;
  Value output = convOp.getResult();
  std::string inputStaticShapeName = emitter.getOrCreateName(output) + emitter.getPPLShapeName(input) + "StaticInput";
  std::string outputStaticShapeName = emitter.getOrCreateName(output) + emitter.getPPLShapeName(output) + "StaticOutput";
  std::string weightStaticShapeName = emitter.getOrCreateName(output) + emitter.getPPLShapeName(weight) + "StaticWeight";
  std::string dilationSuffix = "ConvDilations";
  std::string padsSuffix = "ConvPads";
  std::string stridesSuffix = "ConvStrides";
  auto dilations = convOp.getDilations().has_value() ? convOp.getDilations().value() : NULL;
  int group = (int)convOp.getGroup();
  auto pads = convOp.getPads().has_value() ? convOp.getPads().value() : NULL;
  auto strides = convOp.getStrides().has_value() ? convOp.getStrides().value() : NULL;

  os << "int " << inputStaticShapeName << "[] = ";
  printShape(emitter, input.getType().dyn_cast<TensorType>());
  os << ";\n";

  os << "int " << outputStaticShapeName << "[] = ";
  printShape(emitter, output.getType().dyn_cast<TensorType>());
  os << ";\n";

  os << "int " << weightStaticShapeName << "[] = ";
  printShape(emitter, weight.getType().dyn_cast<TensorType>());
  os << ";\n";

  os << "int " << emitter.getOrCreateName(output) << dilationSuffix << "[] = {";
  if (convOp.getDilations().has_value()) {
    for (auto i : dilations) {
      os << i.dyn_cast<IntegerAttr>().getValue().getRawData()[0] << ", ";
    }
  }
  os << "0};\n";

  os << "int " << emitter.getOrCreateName(output) << padsSuffix << "[] = {";
  if (convOp.getPads().has_value()) {
    for (auto i : pads) {
      os << i.dyn_cast<IntegerAttr>().getValue().getRawData()[0] << ", ";
    }
  }
  os << "0};\n";

  os << "int " << emitter.getOrCreateName(output) << stridesSuffix << "[] = {";
  if (convOp.getStrides().has_value()) {
    for (auto i : strides) {
      os << i.dyn_cast<IntegerAttr>().getValue().getRawData()[0] << ", ";
    }
  }
  os << "0};\n";

  // int conv(
  //     float *input, int *inputShape, int inputRank,
  //     float *output, int *outputShape, int outputRank,
  //     float *weight, int *weightShape, int weightRank,
  //     float *bias, int biasSize,
  //     int *dilations, int dilation_rank
  //     int group, 
  //     int *pads, int pads_rank,
  //     int *strides, int strides_rank
  // );
  os << "conv(";                                                                //conv(
  os << emitter.getOrCreateName(input) << ", ";                                 // input
  os << inputStaticShapeName << ", ";                                           // input shape
  os << (int)input.getType().dyn_cast<TensorType>().getRank() << ", ";          // input rank
  os << emitter.getOrCreateName(output) << ", ";                                // output
  os << outputStaticShapeName << ", ";                                          // output shape
  os << (int)output.getType().dyn_cast<TensorType>().getRank() << ", ";         // output rank
  os << emitter.getOrCreateName(weight) << ", ";                                // weight
  os << weightStaticShapeName << ", ";                                          // weight shape
  os << (int)weight.getType().dyn_cast<TensorType>().getRank() << ", ";         // weight rank
  os << ((convOp.getNumOperands() > 2) ?        
      emitter.getOrCreateName(bias).data() : "NULL") << ", ";             // bias
  os << ((convOp.getNumOperands() > 2) ?
      (int)bias.getType().dyn_cast<TensorType>().getNumElements() : 0) << ", "; // bias size
  os << emitter.getOrCreateName(output) << dilationSuffix << ", ";              // dilations
  os << (convOp.getDilations().has_value() ? dilations.size() : 1) << ", ";     // dilation rank
  os << group << ", ";                                                          // group
  os << emitter.getOrCreateName(output) << padsSuffix << ", ";                  // pads
  os << (convOp.getPads().has_value() ? pads.size() : 1) << ", ";               // pads rank
  os << emitter.getOrCreateName(output) << stridesSuffix << ", ";               // strides
  os << (convOp.getStrides().has_value() ? strides.size() : 1);                 // strides rank
  os << ", " << emitter.getStreamName(output);                                  // stream
  os << ");\n";                                                                 //);
  return success();
}


LogicalResult printOperation(CudaEmitter &emitter, mlir::ONNXConvOp convOp) {
  //TODO: implement conv!!!
  if (useCustomPPL) {
    return printCustomConv(emitter, convOp);
  }
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, func::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();

  return printCallOperation(emitter, operation, callee);
}


//returnOp in onnx is actually assigning returned value to output value correspondly.
//eg. return %0 -> %output_0 := %0
LogicalResult printOperation(CudaEmitter &emitter, mlir::ONNXReturnOp returnOp) {
    //processReturnOp(returnOp);
    raw_indented_ostream &os = emitter.ostream();
    func::FuncOp funcOp = dyn_cast_or_null<func::FuncOp>(returnOp->getBlock()->getParent()->getParentOp());
    FunctionType funcType = funcOp.getFunctionType();
    ArrayAttr resAttrs = funcOp.getResAttrsAttr();
    auto returnOprands = returnOp.getOperands();
 
    if (returnOp.getNumOperands() != funcType.getNumResults()) {
      return returnOp.emitError("return value number does not match the function output number!");
    }

    while (emitter.hasCudaEvent()) {
        auto eventName = emitter.popCudaEvent();
        os << "cudaEventSynchronize(" << eventName << "Event);\n";
        os << "cudaEventDestroy(" << eventName <<"Event);\n";
    }

    for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
      if (resAttrs) {
        DictionaryAttr dictAttrs = llvm::dyn_cast<DictionaryAttr>(resAttrs[i]);
        if (dictAttrs && dictAttrs.contains("onnx.name")) {
          if (auto tType = returnOprands[i].getType().dyn_cast<TensorType>()) {
            os << "cudaMemcpy(";
            os << "func_output_" << dictAttrs.getNamed("onnx.name")
                            .value()
                            .getValue()
                            .cast<StringAttr>().strref();
            os << ", ";
            os << emitter.getOrCreateName(returnOprands[i]) << ", ";
            os << tType.getNumElements();
            os << " * sizeof(";
            if (failed(emitter.emitType(returnOprands[i].getLoc(), tType.getElementType()))) {
              return returnOp.emitError("emit return value type failed!");
            } 
            os << "), cudaMemcpyDeviceToDevice);\n";
            Value value = returnOp.getOperand(0);
            if(failed(emitter.emitFree(value))) {
              return failure();
            }
          } else {
            os << "func_output_" << dictAttrs.getNamed("onnx.name")
                            .value()
                            .getValue()
                            .cast<StringAttr>().strref();
            //os << " /*output[" << i << "]*/";
            os << " = ";
            if (!emitter.hasValueInScope(returnOprands[i])) {
              return returnOp.emitError("return undefined value!");
            }
            os << emitter.getOrCreateName(returnOprands[i]);
            os << ";\n";
          }
        }
      }
    }
    if (enableTiming) {
      os << "std::cout << \"total compute time: \" << ms_total << \"ms\" << std::endl;\n"; 
    }
    os << "return;";
    return success();
}

//returnOp in onnx is actually assigning returned value to output value correspondly.
//eg. return %0 -> %output_0 := %0
LogicalResult printOperation(CudaEmitter &emitter, func::ReturnOp returnOp) {
    //processReturnOp(returnOp);
    raw_indented_ostream &os = emitter.ostream();
    func::FuncOp funcOp = dyn_cast_or_null<func::FuncOp>(returnOp->getBlock()->getParent()->getParentOp());
    FunctionType funcType = funcOp.getFunctionType();
    ArrayAttr resAttrs = funcOp.getResAttrsAttr();
    auto returnOprands = returnOp.getOperands();
 
    if (returnOp.getNumOperands() != funcType.getNumResults()) {
      return returnOp.emitError("return value number does not match the function output number!");
    }

    for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
      if (resAttrs) {
        DictionaryAttr dictAttrs = llvm::dyn_cast<DictionaryAttr>(resAttrs[i]);
        if (dictAttrs && dictAttrs.contains("onnx.name")) {
          if (auto tType = returnOprands[i].getType().dyn_cast<TensorType>()) {
            os << "cudaMemcpy(";
            os << "func_output_" << dictAttrs.getNamed("onnx.name")
                            .value()
                            .getValue()
                            .cast<StringAttr>().strref();
            os << ", ";
            os << emitter.getOrCreateName(returnOprands[i]) << ", ";
            os << tType.getNumElements();
            os << " * sizeof(";
            if (failed(emitter.emitType(returnOprands[i].getLoc(), tType.getElementType()))) {
              return returnOp.emitError("emit return value type failed!");
            } 
            os << "), cudaMemcpyDeviceToDevice);\n";
            Value value = returnOp.getOperand(0);
            if(failed(emitter.emitFree(value))) {
              return failure();
            }
          } else {
            os << "func_output_" << dictAttrs.getNamed("onnx.name")
                            .value()
                            .getValue()
                            .cast<StringAttr>().strref();
            //os << " /*output[" << i << "]*/";
            os << " = ";
            if (!emitter.hasValueInScope(returnOprands[i])) {
              return returnOp.emitError("return undefined value!");
            }
            os << emitter.getOrCreateName(returnOprands[i]);
            os << ";\n";
          }
        }
      }
    }

    os << "return;";
    return success();
}

LogicalResult printFuncGlobalResorce(CudaEmitter &emitter, func::FuncOp funcOp) {
  return failure();
}

LogicalResult printFuncPreProcess(CudaEmitter &emitter, func::FuncOp funcOp) {
  raw_indented_ostream &os = emitter.ostream();



  os << "__host__ void " << funcOp.getName().str() << "PreProcess() {\n";

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, func::FuncOp funcOp) {
  raw_indented_ostream &os = emitter.ostream();
  FunctionType funcType = funcOp.getFunctionType();
  auto outputs = funcType.getResults();
  ArrayAttr resAttrs = funcOp.getResAttrsAttr();

  if (failed(emitter.trackValueLifetime(funcOp))) {
    return failure();
  }
#if 0
  funcOp.walk([&](Operation *op){
      raw_indented_ostream &os = emitter.ostream();
    if (op->getNumResults()) {
      for (auto i : op->getResults()) {
        if (emitter.getValueFirstUse(i)) {
          os << emitter.getOrCreateName(i) << " fisrt use at ";
          os << emitter.getOrCreateName(i) << " = " << op->getName() << "(";
          for (auto j : emitter.getValueFirstUse(i)->getOperands()) {
            os << emitter.getOrCreateName(j) << ", ";
          }
          os << ");\n";
        }
        if (emitter.getValueLastUse(i)) {
          os << emitter.getOrCreateName(i) << " last use at ";
          os << emitter.getOrCreateName(i) << " = " << op->getName() << "(";
          for (auto j : emitter.getValueLastUse(i)->getOperands()) {
            os << emitter.getOrCreateName(j) << ", ";
          }
          os << ");\n";
        }
        os << "\n";
      }
    }
  });
#endif
  os << "__host__ void " << funcOp.getName().str() << "(";
  for (auto arg : funcOp.getArguments()) {
    if(failed(emitter.emitType(funcOp.getLoc(), arg.getType()))) {
      return funcOp.emitOpError("func args emit failed!");
    }
    os << emitter.getOrCreateName(arg) << ", ";
  }
  for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
    if (i) {
      os << ", ";
    }
    if (failed(emitter.emitType(funcOp.getLoc(), outputs[i]))) {
      return funcOp.emitOpError("func args emit failed!");
    }
    if (resAttrs) {
      DictionaryAttr dictAttrs = llvm::dyn_cast<DictionaryAttr>(resAttrs[i]);
      if (dictAttrs && dictAttrs.contains("onnx.name")) {
        os << "func_output_" << dictAttrs.getNamed("onnx.name")
                        .value()
                        .getValue()
                        .cast<StringAttr>().strref();
        //os << " /*output[" << i << "]*/";
      }
    }
  }
  os << ") {";
  os.indent();
  os << "\n";

  INFO("initialize ms for timing, if enabled")
  if (enableTiming) {
    os << "float ms = 0.f;\n";
    os << "float ms_total = 0.f;\n";
  }

  INFO("prepare some re-use var for ops")
  if (emitter.hasPplOp("onnx.Concat") ||
      emitter.hasPplOp("onnx.SplitV13")
  ) {
    os << "std::vector<ppl::common::TensorShape> pplInputShapes;\n";
  }

  if (emitter.hasPplOp("onnx.Transpose")) {

  }

  INFO("emit pplshape decl for func args")
  for (auto i : funcOp.getArguments()) {
    if (failed(emitter.emitPPLShapeDeclaration(i))) {
      return funcOp.emitOpError("emit func arg ppl shape decl failed!");
    }
  }
  os << "\n";

  // 遍历当前操作的所有子区域（如果有）
  for (Region &region : funcOp->getRegions()) {
    // 遍历每个区域中的所有块
    for (Block &block : region) {
      // 对块中的每个操作递归调用此函数
      for (Operation &childOp : block) {
        if (failed(emitter.emitOperation(childOp, false))) {
          return funcOp.emitOpError("func body emit failed!");
        }
      }
    }
  }
  os.unindent();
  os << "}\n";

  return success();
}

LogicalResult printConstantAsm(CudaEmitter &emitter, mlir::ONNXConstantOp constOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = constOp.getResult();
  std::string constName = emitter.getConstantName(res);
  size_t lineBreaker = 0;

  os << "\"" << emitter.getConstantName(constOp.getResult()) << ":\\n\"\n";

  assert(constOp.getValue().has_value() && "Value is not set");

  bool showConstContent = true;
  if (auto tensorAttr = constOp.getValueAttr().dyn_cast<DisposableElementsAttr>()) {
    if (auto tType = res.getType().dyn_cast<TensorType>()) {
      auto eType = tType.getElementType();
      if (eType.isa<Float16Type>()) {
        os << "\".short ";
        auto t =  tensorAttr.getArray<float_16>().get();
        for (auto i : t) {
          os << "0X";
          os.write_hex(i.bitcastToUInt());
          if (!(lineBreaker % 1200)) {
            os << "\\n\"\n";
            os << "\".short ";
            lineBreaker = 0;
          } else {
            os << ", ";
          }
          lineBreaker++;
          if (!showConstContent) {
            break;
          }
        }
      } else if (eType.isa<Float32Type>()) {
        os << "\".float ";
        auto t =  tensorAttr.getArray<float>().get();
        for (auto i : t) {
          os << i;
          if (!(lineBreaker % 600)) {
            os << "\\n\"\n";
            os << "\".float ";
            lineBreaker = 0;
          } else {
            os << ", ";
          }
          lineBreaker++;
          if (!showConstContent) {
            break;
          }
        }
      } else if (eType.isa<IntegerType>()) {
        auto iType = eType.dyn_cast<IntegerType>();
        if (iType.getWidth() == 64) {
          os << "\".quad ";
          auto t =  tensorAttr.getArray<int64_t>().get();
          for (auto i : t) {
            os << "0X";
            os.write_hex(i);
            if (!(lineBreaker % 1200)) {
              os << "\\n\"\n";
              os << "\".quad ";
              lineBreaker = 0;
            } else {
              os << ", ";
            }
            lineBreaker++;
            if (!showConstContent) {
              break;
            }
          }
        } else if (iType.getWidth() == 32) {
          os << "\".word ";
          auto t =  tensorAttr.getArray<int32_t>().get();
          for(auto i : t) { os << i << ", "; if(!showConstContent) { break; }}
        } else if (iType.getWidth() == 16) {
          auto t =  tensorAttr.getArray<int16_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; if(!showConstContent) { break; }}
        } else if (iType.getWidth() == 8) {
          auto t =  tensorAttr.getArray<int8_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << (int)i << ", "; if(!showConstContent) { break; }}
        } else {
          os << "WTF: ??? " << res.getLoc() << "\n"; 
        }
      } else {
        os << "WTF: ??? " << res.getLoc() << "\n";
      }
      os << "0";
      os << "\\n\"\n";
    }
  }

  return success();
}

LogicalResult printConstantAsm(CudaEmitter &emitter, ModuleOp moduleOp) {
  raw_indented_ostream &os = emitter.ostream();

  os << "asm (\n";
  os << "\".section .czw_onnx_const, \\\"a\\\"\\n\"\n";
  os << "\".align 8\\n\"\n";

  //traverse all graph for constant
  for (Operation &op : moduleOp) {
    if (auto func = dyn_cast_if_present<func::FuncOp>(op)) {
      func.walk([&](Operation *op){
        if (auto constOp = dyn_cast_if_present<mlir::ONNXConstantOp>(op)) {
          if (failed(printConstantAsm(emitter, constOp))) {

          }
        }
      });
    }
  }

  os << "\".text\\n\"\n";
  os << ");\n";
  return success();
}

void CudaEmitter::collectPplInc(CudaEmitter &emitter, ModuleOp moduleOp) {
  moduleOp->walk([&](Operation *op){
    if (op->getDialect()->getNamespace() == "onnx") {
      insertPplOp(op->getName().getStringRef());
    }
  });
}

void CudaEmitter::emitInclude(StringRef name, bool isLocal) {
  os << "#include ";
  os << (isLocal ? "\"" : "<");
  os << name;
  os << (isLocal ? "\"" : ">") << "\n";
} 


void CudaEmitter::printPplInc(CudaEmitter &emitter) {
  bool isLocal = true;
  StringRef prefix = "cudakernel/";
  if (hasPplOp("onnx.Add") || 
      hasPplOp("onnx.Mul") ||
      hasPplOp("onnx.Sub") ||
      hasPplOp("onnx.Div") ||
      hasPplOp("onnx.Pow")
    ) { emitter.emitInclude(prefix.str().append("arithmetic/arithmetic.h"), isLocal); }
  if (hasPplOp("onnx.Abs") ||
      hasPplOp("onnx.Sigmoid")
    ) { emitter.emitInclude(prefix.str().append("unary/unary.h"), isLocal); }
  if (hasPplOp("onnx.Custom")) { emitter.emitInclude(prefix.str().append("unary/swish.h"), isLocal); }
  if (hasPplOp("onnx.Concat")) { emitter.emitInclude(prefix.str().append("memory/concat.h"), isLocal); }
  if (hasPplOp("onnx.MaxPoolSingleOut")) { emitter.emitInclude(prefix.str().append("nn/pooling_max.h"), isLocal);}
  if (hasPplOp("onnx.Reshape")) { emitter.emitInclude(prefix.str().append("memory/reshape.h"), isLocal);}
  if (hasPplOp("onnx.ResizeV13")) { emitter.emitInclude(prefix.str().append("nn/resize.h"), isLocal);}
  if (hasPplOp("onnx.SplitV13")) { emitter.emitInclude(prefix.str().append("memory/split.h"), isLocal);}
  if (hasPplOp("onnx.Transpose")) { emitter.emitInclude(prefix.str().append("memory/transpose.h"), isLocal);}
  if (hasPplOp("onnx.Conv") && useCustomPPL) { emitter.emitInclude("conv2d.h", isLocal);}
  os << "\n";
}

void printHelperFunction(CudaEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  os << "__host__ int **createPPLDims(std::vector<ppl::common::TensorShape> &shapes) {\n";
  os << "  int **res = (int **)malloc(sizeof(*res) * shapes.size());\n";
  os << "  for (auto i = 0; i < shapes.size(); i++) {\n";
  os << "    int *dims = (int *)malloc(sizeof(*dims) *shapes[i].GetDimCount());\n";
  os << "    const int64_t *shapeDims = shapes[i].GetDims();\n";
  os << "    for (auto j = 0; j < shapes[i].GetDimCount(); j++) {\n";
  os << "      dims[j] = (int32_t)shapeDims[j];\n";
  os << "    }\n";
  os << "    res[i] = dims;\n";
  os << "  }\n";
  os << "\n";
  os << "  return res;\n";
  os << "}\n\n";
  os << "__host__ void destroyPPLDims(int **ptr, int numShape) {\n";
  os << "  for (auto i = 0; i < numShape; i++) {\n";
  os << "    free(ptr[i]);\n";
  os << "  }\n";
  os << "  free(ptr);\n";
  os << "}\n\n\n";

  os << "__host__ const int64_t **createPPLDimsI64(std::vector<ppl::common::TensorShape> &shapes) {\n";
  os << "  const int64_t **res = (const int64_t **)malloc(sizeof(*res) * shapes.size());\n";
  os << "  for (auto i = 0; i < shapes.size(); i++) {\n";
  os << "    int64_t *dims = (int64_t *)malloc(sizeof(*dims) *shapes[i].GetDimCount());\n";
  os << "    const int64_t *shapeDims = shapes[i].GetDims();\n";
  os << "    for (auto j = 0; j < shapes[i].GetDimCount(); j++) {\n";
  os << "      dims[j] = (int64_t)shapeDims[j];\n";
  os << "    }\n";
  os << "    res[i] = dims;\n";
  os << "  }\n";
  os << "\n";
  os << "  return res;\n";
  os << "}\n";

  os << "__host__ void destroyPPLDimsI64(const int64_t **ptr, int numShape) {\n";
  os << "  for (auto i = 0; i < numShape; i++) {\n";
  os << "    free((void *)ptr[i]);\n";
  os << "  }\n";
  os << "  free(ptr);\n";
  os << "}\n\n\n";
}

LogicalResult printOperation(CudaEmitter &emitter, ModuleOp moduleOp) {
  printFixedCode(emitter);
  emitter.collectPplInc(emitter, moduleOp);
  emitter.printPplInc(emitter);
  printHelperFunction(emitter);

  if (failed(printConstantAsm(emitter, moduleOp))) {
    return failure();
  }
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXCustomOp customOp) {
    raw_indented_ostream &os = emitter.ostream();

    // Extract attributes from ONNXCustomOp
    auto domainName = customOp->getAttrOfType<StringAttr>("domain_name").getValue();
    auto functionName = customOp->getAttrOfType<StringAttr>("function_name").getValue();

    // TODO: maybe there are a lot self-define operations?
    // Verify the domain name and function name to identify the CustomOp
    if (domainName != "com.metax-tech" || functionName != "Swish") {
        return failure(); // CustomOp doesn't match expected criteria
    }

    // Extract operands and results
    auto operands = customOp.getOperands();
    auto results = customOp.getResults();

    // Swish has only one operand and one result, right?
    auto inputTensor = operands[0];
    auto outputTensor = results[0];

    // Print information about the CustomOp
    os << "PPLCUDAUnarySwishForwardImp(";                           // PPLCUDAUnarySwishForwardImp(
    os << emitter.getStreamName(outputTensor) << ", ";              // cudaStream_t stream,
    os << "&" << emitter.getPPLShapeName(inputTensor) << ", ";      // const ppl::common::TensorShape* input_shape,
    os << emitter.getOrCreateName(inputTensor) << ", ";             // const void* input,
    os << "&" << emitter.getPPLShapeName(outputTensor) << ", ";     // const ppl::common::TensorShape* output_shape,
    os << emitter.getOrCreateName(outputTensor) << ", ";            // void* output,
    os << "0.f, ";                                                  // float beta, actually not used
    os << "1.f, ";                                                  // float in_scale,
    os << "1.f";                                                    // float out_scale
    os << ");\n";

    return success();
}

std::string CudaEmitter::getStreamName(Value value) {
  return  enableStreamAndEvent ?
      getOrCreateName(value.getDefiningOp()->getResult(0)) + "Stream"
    : "0";
}

std::string CudaEmitter::getEventName(Value value) {
  assert(value.getDefiningOp());
  return getOrCreateName(value.getDefiningOp()->getResult(0)) + "Event";
}

std::string CudaEmitter::getConstantName(Value value) {
  assert(value.getDefiningOp());
  assert(isa<mlir::ONNXConstantOp>(value.getDefiningOp()));
  auto fop = value.getDefiningOp();
  while (fop && !isa<func::FuncOp>(fop)) {
    fop = fop->getParentOp();
  }

  return (fop ? dyn_cast_if_present<func::FuncOp>(fop).getName().str() : "NOFUNC") + "_" + getOrCreateName(value) + "Constant";
}

LogicalResult CudaEmitter::emitONNXPreOp (Operation &op) {
  //if op has result(all onnx op  has at least one output, so just a guard here)
  if (op.getNumResults()) {
    INFO("SSA var declaration and shape decl if SSA is TensorType")
    for (auto res : op.getResults()) {
      if (failed(emitDeclaration(res))) {
        return op.emitOpError("unable to declare ssa var.");
      }
      if (auto tType = dyn_cast_if_present<TensorType>(res.getType())) {
        if (failed(emitDeviceMalloc(res))) {
          return op.emitOpError("unable to malloc ssa var.");
        }
        if(failed(emitPPLShapeDeclaration(res))) {
          return op.emitError("failed to emit ppl shape declaration!");
        }
      }
    }

    //Name stream and event after res[0]. Declare and init stream and event for every ONNX op.
    if (enableStreamAndEvent) {
      INFO("create stream and event, naming after result[0].")
      Value res0 = op.getResult(0);
      os << "cudaStream_t " << getStreamName(res0) << ";\n";
      os << "cudaStreamCreate(&" << getStreamName(res0) << ");\n";
      os << "cudaEvent_t " << getEventName(res0) << ";\n";
      os << "cudaEventCreate(&" << getEventName(res0) << ");\n";
      insertCudaEvent(res0, op.getNumResults());

      INFO("Wait for events for every operand.")
      if (op.getNumOperands()) {
        for (auto operand : op.getOperands()) {
          if (NULL == operand.getDefiningOp()) { continue; }
          os << "cudaStreamWaitEvent(" << getStreamName(res0) << ", " << getEventName(operand) << ", 0);\n"; 
        }
      }

      if (enableTiming) {
        os << "cudaEvent_t " << getEventName(res0) << "Start;\n";
        os << "cudaEventCreate(&" << getEventName(res0) << "Start);\n";
        os << "cudaEventRecord(" << getEventName(res0) << "Start, " << getStreamName(res0) << ");\n";  
      }
    }
  }

  return success();
}

LogicalResult CudaEmitter::emitONNXPostOp(Operation &op) {
  INFO("Post proccess")
  if (enableStreamAndEvent) {
    INFO("1. emit event record")
    os << "cudaEventRecord(" << getEventName(op.getResult(0)) << ", " << getStreamName(op.getResult(0)) << ");\n";
  }
  INFO("2. destroy first use operand`s stream (func args do not have definingOp and do not need streamdestroy)")
  INFO("3. free last use for every operands")
  if (op.getNumOperands()) {
    for (auto operand : op.getOperands()) {
      if (enableStreamAndEvent && operand.getDefiningOp() && getValueFirstUse(operand) == (&op)) {
        os << "cudaStreamDestroy(" << getStreamName(operand) << ");\n";
      }
      if (getValueLastUse(operand) == (&op)) {
        if (failed(emitFree(operand))) {
          return failure();
        }
        if (enableStreamAndEvent && operand.getDefiningOp()) {
          if (numRefCudaEvent(operand) == 1) {
            os << "cudaEventSynchronize(" << getEventName(operand) << ");\n";
            if (enableTiming) {
              os << "cudaEventElapsedTime(&ms, " << getEventName(operand) << "Start, ";
              os << getEventName(operand) << ");\n";
              os << "std::cout << \"" << getOrCreateName(operand) << ", " << operand.getDefiningOp()->getName();
              os << ", \" << ms << std::endl;\n"; 
            }
            os << "cudaEventDestroy(" << getEventName(operand) <<");\n";
            if (operand.getDefiningOp()->getName().getStringRef() != "onnx.Constant")
               os << "ms_total+=ms;\n";
          }
          dropCudaEvent(operand);
        }
      }
    }
  }
  os << "\n\n";

  return success();
}

LogicalResult CudaEmitter::emitDeclaration(Value &value) {
  INFO(value.getLoc());
  if (failed(this->emitType(value.getLoc(), value.getType()))) {
    return failure();
  }

  os << getOrCreateName(value) << ";";
  os << "\n";

  return success();
}

LogicalResult CudaEmitter::emitDeviceMalloc(Value &value) {
  INFO(value.getLoc());

  if (auto tType = value.getType().dyn_cast<TensorType>()) {
    os << "cudaMalloc((void**)&" << getOrCreateName(value) << ", "<< tType.getNumElements();
    os << " * sizeof(";
    if (failed(emitType(value.getLoc(), tType.getElementType()))) {
      return failure();
    }
    os << "));\n";
    return success();
  }
  return failure();
}

LogicalResult CudaEmitter::emitFree(Value &value) {
  if (auto tType = value.getType().dyn_cast<TensorType>()) {
    os << "cudaFree(" << getOrCreateName(value) << ");\n";
  }

  return success();
}


LogicalResult CudaEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // CF ops.
          //.Case<cf::BranchOp, cf::CondBranchOp>(
          //    [&](auto op) { return printOperation(*this, op); })
          // ONNX ops.
          .Case<mlir::ONNXAbsOp,
                mlir::ONNXAddOp, mlir::ONNXMulOp, mlir::ONNXDivOp, mlir::ONNXSubOp,
                mlir::ONNXConcatOp, mlir::ONNXConstantOp, mlir::ONNXMaxPoolSingleOutOp,
                mlir::ONNXPowOp, mlir::ONNXReshapeOp, mlir::ONNXResizeV13Op,
                mlir::ONNXSigmoidOp, mlir::ONNXSplitV13Op, mlir::ONNXTransposeOp,
                mlir::ONNXConvOp,
                mlir::ONNXCustomOp
                >(
              [&](auto op) {
                Operation *opop = op.getOperation();
                INFO(op.getLoc())
                if (failed(emitONNXPreOp(*opop)))         { return failure(); }
                if (failed(printOperation(*this, op)))    { return failure(); }
                if (failed(emitONNXPostOp(*opop)))        { return failure(); }
                return success();
          })
          .Case<mlir::ONNXNoneOp>(
            [&](auto op) {
              Operation *opop = op.getOperation();
              if (failed(emitONNXPreOp(*opop)))         { return failure(); }
              if (failed(emitONNXPostOp(*opop)))        { return failure(); }
              return success(); 
          })
          // Func ops.
          .Case<func::CallOp, func::FuncOp, func::ReturnOp, mlir::ONNXReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          //.Case<mlir::LiteralOp>([&](auto op) { return success(); })
          // ignore entry point, we will call func somewhere else
          .Case<mlir::ONNXEntryPointOp>([&](auto op) { return success(); })
          .Default([&](Operation *) {
            //fixme: temp test
            //return op.emitOpError("unable to find printer for op");
            if (failed(emitONNXPreOp(op)))         { return failure(); }
            os << op.getName() << " PLACEHOLDER, not impoemented yet.\n"; return success();
            if (failed(emitONNXPostOp(op)))        { return failure(); }

          });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");

  return success();
}

LogicalResult translateToCuda(Operation *op, raw_ostream &os,
                                    bool declareVariablesAtTop) {
  CudaEmitter emitter(os, declareVariablesAtTop);
  CudaEmitter::Scope scope(emitter);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}

} //namespace onnx_mlir

namespace mlir {

//===----------------------------------------------------------------------===//
// Cuda registration
//===----------------------------------------------------------------------===//

void registerToCudaTranslation() {
  static llvm::cl::opt<bool> declareVariablesAtTop(
      "declare-variables-at-top",
      llvm::cl::desc("Declare variables at top when emitting C/C++"),
      llvm::cl::init(false));

  TranslateFromMLIRRegistration reg(
      "onnxir-to-cuda", "translate from onnxir to cuda",
      [](Operation *op, raw_ostream &output) {
        return onnx_mlir::translateToCuda(
            op, output,
            /*declareVariablesAtTop=*/declareVariablesAtTop);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<cf::ControlFlowDialect,
                        mlir::ONNXDialect,
                        func::FuncDialect,
                        math::MathDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}

} // namespace mlir


