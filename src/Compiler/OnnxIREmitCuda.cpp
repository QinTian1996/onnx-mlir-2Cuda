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
#include "../Dialect/Mlir/DialectBuilder.hpp"
#include <llvm/IR/DerivedTypes.h>



#define DEBUG_TYPE "translate-to-cuda"

using namespace mlir;
using llvm::formatv;

#define INFO(info) os << "//INFO: " << info << "\n";


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
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    llvm::ScopedHashTableScope<Value, Operation *> lifetimeTrackerFirstScope;
    llvm::ScopedHashTableScope<Value, Operation *> lifetimeTrackerLastScope;
    CudaEmitter &emitter;
  };

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

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

  //Push/Pop cudaEvent record.
  void pushCudaEvent(Value value) { eventRecord.push(value); };
  Value popCudaEvent(void) { Value v = eventRecord.top(); eventRecord.pop(); return v; };
  bool hasCudaEvent(void) { return !eventRecord.empty(); };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
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
  std::stack<Value> eventRecord;

  /// Collect all onnx op using ppl.cv to print include lines
  /// eg. mlir::ONNXAbsOp -> #include <ppl/cv/cuda/Abs.h>
  std::set<StringRef> onnx_op_types;

  /// Track first and last use of Value
  LifeTimeTracker valueFisrtUseTracker;
  LifeTimeTracker valueLastUseTracker;

public:
  LogicalResult emitPPLShapeDeclaration(Value &value);
  std::string getPPLShapeName(Value &value) {
    return getOrCreateName(value).str() + "Shape";
  };

private:
  std::string pplCommonPrefix = "ppl::common::";
  LogicalResult emitPPLType(Type type);

};

CudaEmitter::CudaEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef CudaEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
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
  if (!valueFisrtUseTracker.count(value)) {
    return NULL;
  }
  return valueFisrtUseTracker.lookup(value);
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
  os << "TensorShape: [";
  for (auto i = 0; i < tType.getRank(); i++) {
    os << ( (i == 0)? "" : ", ") << (tType.isDynamicDim(i) ? "Dyn" : std::to_string(tType.getDimSize(i)));
  }
  os << "]\n";
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
  ONNXConstantOp constOp;

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
  os << "//A "; printShape(emitter, tTypeA);
  os << "//B "; printShape(emitter, tTypeB);
  os << "//C "; printShape(emitter, tTypeC);
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
  os << "//A "; printShape(emitter, tTypeA);
  os << "//B "; printShape(emitter, tTypeB);
  os << "//C "; printShape(emitter, tTypeC);
  os.unindent();

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXMaxPoolSingleOutOp maxPoolSingleOutOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = maxPoolSingleOutOp.getResult();
  Value inp = maxPoolSingleOutOp.getOperand();

  int kernelH = maxPoolSingleOutOp.getKernelShape()[0].dyn_cast<IntegerAttr>().getValue().getRawData()[0];
  int kernelW = maxPoolSingleOutOp.getKernelShape()[0].dyn_cast<IntegerAttr>().getValue().getRawData()[0];
  int strideH = maxPoolSingleOutOp.getStrides().has_value() ? maxPoolSingleOutOp.getStrides().value()[0].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 1;
  int strideW = maxPoolSingleOutOp.getStrides().has_value() ? maxPoolSingleOutOp.getStrides().value()[1].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 1;;
  int padH    = maxPoolSingleOutOp.getPads().has_value() ? maxPoolSingleOutOp.getPads().value()[0].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 0;
  int padW    = maxPoolSingleOutOp.getPads().has_value() ? maxPoolSingleOutOp.getPads().value()[1].dyn_cast<IntegerAttr>().getValue().getRawData()[0] : 0;
  
  os << "ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(";  //ppl::common::RetCode PPLCUDAMaxPoolingForwardImp(
  os << emitter.getStreamName(res) << ", ";                   //  cudaStream_t stream,
  os << "&" << emitter.getPPLShapeName(inp) << ", ";          //  ppl::common::TensorShape* input_shape,
  os << emitter.getOrCreateName(inp) << ", ";                 //  const void* input,
  os << "&" << emitter.getPPLShapeName(res) << ", ";          //  ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(res) << ", ";                 //  void* output,
  os << kernelH << ", ";                                      //  int kernel_height,
  os << kernelW << ", ";                                      //  int kernel_width,
  os << strideH << ", ";                                      //  int stride_height,
  os << strideW << ", ";                                      //  int stride_width,
  os << padH << ", ";                                         //  int padding_height,
  os << padW << ", ";                                         //  int padding_width,
  os << 1.0f << ", ";                                         //  float in_scale,
  os << 1.0f << ");\n";                                       //  float out_scale);

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXConcatOp concatOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = concatOp.getResult();
  auto operands = concatOp.getOperands();
 
  INFO("collect input_dims and input_padded_dims, since ONNX.concat does not have pad attr, these 2 same.")
  INFO("need to convert int64 * dims to int32 *")

  // collect shapes
  os << "pplConcatInputShapes.clear();\n";
  for (auto i : operands) {
    os << "pplConcatInputShapes.push_back(" << emitter.getPPLShapeName(i) << ");\n";
  }

  os <<  "int **" << emitter.getOrCreateName(res) << "PPLConcatInputDims = createPPLDims(pplConcatInputShapes);\n";

  INFO("collect inputs.");
  os << "void *" << emitter.getOrCreateName(res) << "ONNXConcatOpInputs[] = {";
  for (auto i : operands) {
    os << emitter.getOrCreateName(i) << ", ";
  }
  os << "};\n";

  std::string stream = emitter.getStreamName(res);
  auto axis = concatOp.getAxis();
  auto numInput = concatOp.getNumOperands();

  os << "PPLCUDAConcatForwardImp(";                                       //  ppl::common::RetCode PPLCUDAConcatForwardImp(
  os << stream                                                    << ", ";//      cudaStream_t stream,
  os << axis                                                      << ", ";//      int axis,
  os << numInput                                                  << ", ";//      int num_inputs,
  os << emitter.getOrCreateName(res) << "PPLConcatInputDims"      << ", ";//      int* input_dims[],
  os << emitter.getOrCreateName(res) << "PPLConcatInputDims"      << ", ";//      int* input_padded_dims[],
  os << emitter.getOrCreateName(res) << "ONNXConcatOpInputs"      << ", ";//      const void* inputs[],
  os << emitter.getPPLShapeName(res)                              << ", ";//      ppl::common::TensorShape* output_shape,
  os << emitter.getOrCreateName(res)                              << ", ";//      void* output,
  os << "0"                                                       << ");";//      int mask = 0);
  os << "\n";

  os << "destroyPPLDims(" << emitter.getOrCreateName(res) << "PPLConcatInputDims";
  os << ", pplConcatInputShapes.size());\n";
  os << "pplConcatInputShapes.clear();\n";

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXConstantOp constantOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value res = constantOp.getResult();
  size_t size = 0;
  Type type = res.getType();

  if (constantOp.getSparseValue().has_value()) {
    return constantOp.emitError("Only support dense values at this time");
  }
  assert(constantOp.getValue().has_value() && "Value is not set");

  //FIXME: enable constant print
  bool showConstContent = false;
  if(!showConstContent) { return success(); }
  if (auto tensorAttr = constantOp.getValueAttr().dyn_cast<DisposableElementsAttr>()) {
    if (auto tType = res.getType().dyn_cast<TensorType>()) {
      auto eType = tType.getElementType();
      if (failed(emitter.emitType(res.getLoc(), eType))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(res) << "constant[] = {";
      if (eType.isa<Float16Type>()) {
        auto t =  tensorAttr.getArray<float_16>();
        auto t1 = t.get();
        for(auto i : t1) { os << i.toFloat() << ", "; }
        size = t1.size() * sizeof(float_16);
      }
      else if (eType.isa<Float32Type>()) {
        auto t =  tensorAttr.getArray<float>();
        auto t1 = t.get();
        for(auto i : t1) { os << i << ", "; }
        size = t1.size() * sizeof(float);
      } else if (eType.isa<IntegerType>()) {
        auto iType = eType.dyn_cast<IntegerType>();
        if (iType.getWidth() == 64) {
          auto t =  tensorAttr.getArray<int64_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; }
          size = t1.size() * sizeof(int64_t);
        } else if (iType.getWidth() == 32) {
          auto t =  tensorAttr.getArray<int32_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; }
          size = t1.size() * sizeof(int);
        } else if (iType.getWidth() == 16) {
          auto t =  tensorAttr.getArray<int16_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << i << ", "; }
          size = t1.size() * sizeof(int16_t);
        } else if (iType.getWidth() == 8) {
          auto t =  tensorAttr.getArray<int8_t>();
          auto t1 = t.get();
          for(auto i : t1) { os << (int)i << ", "; }
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

          if (failed(emitter.emitType(res.getLoc(), iType))) {
            return constantOp.emitError("emit dense ints const value failed!");
          }
          os << " " << emitter.getOrCreateName(res) << "Constant[] = {";
          for (auto i : valueRange) {
            os << *i.getValue().getRawData() << ", ";
          }
          os <<"};\n";
          size = valueRange.size() * iType.getWidth() / 8;
        }
      } else if (auto floatDenseAttr = tensorAttr.dyn_cast_or_null<DenseFPElementsAttr>()) {
        auto valueRange = floatDenseAttr.getValues<FloatAttr>();
        if (auto fType = eType.dyn_cast<FloatType>()) {
          if (failed(emitter.emitType(res.getLoc(), fType))) {
            return constantOp.emitError("emit dense ints const value failed!");
          }
          os << " " << emitter.getOrCreateName(res) << "Constant[] = {";
          for (auto i : valueRange) {
            os << i.getValue().convertToFloat() << ", ";
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
    os << "float " << emitter.getOrCreateName(res) << "Constant[] = {";
    for (auto i : fs.getValue()) {
      os << i.cast<FloatAttr>().getValue().convertToFloat() << ", ";
    }
    os << "};\n";
    size = sizeof(float) * fs.size();
  } else if (constantOp.getValueInt().has_value()) {
    int i = constantOp.getValueInt().value();
    os << emitter.getOrCreateName(res) << " = " << i << ";\n";
  } else if (constantOp.getValueInts().has_value()) {
    auto fs = constantOp.getValueInts().value().getValue();
    os << "int " << emitter.getOrCreateName(res) << "Constant[] = {";
    for (auto i : fs) {
      os << i.dyn_cast<IntegerAttr>().getValue() << ", ";
    }
    os << "};\n";
    size = sizeof(int) * fs.size();
  } else if (constantOp.getValueString().has_value()) {
    os << "char *" << emitter.getOrCreateName(res) << "Constant[] = \"";
    os <<  constantOp.getValueString().value() << "\"\n";
    size = constantOp.getValueString().value().size();
  } else if (constantOp.getValueStrings().has_value()) {
    return constantOp.emitError("wtf wtf wtf!");
  } else {
    llvm::errs() << "constant type : " << res.getType() << "\n";
    //llvm::errs() << "type : " << constantOp.getValueAttr() << "\n";
    return constantOp.emitError("string list and other constant type not supported yet!");
  }

  if (size) {
    os << "cudaMemcpyAsync(";
    os << emitter.getOrCreateName(res) << ", "; //dst
    os << emitter.getOrCreateName(res) << "Constant" << ", "; //src
    os << size << ", ";
    os << "cudaMemcpyHostToDevice, ";
    os << emitter.getStreamName(res) << ");\n";
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

    while (emitter.hasCudaEvent()) {
      Value value = emitter.popCudaEvent();
      os << "cudaEventSynchronize(" << emitter.getEventName(value) << ");\n";
      os << "cudaEventDestroy(" << emitter.getEventName(value) << ");\n";
    }

    for (unsigned int i = 0; i < funcType.getNumResults(); i++) {
      if (resAttrs) {
        DictionaryAttr dictAttrs = llvm::dyn_cast<DictionaryAttr>(resAttrs[i]);
        if (dictAttrs && dictAttrs.contains("onnx.name")) {
          //cudaMemcpy
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

    //Only destroy event of SSA which are first res of its definingOp to prevent repeat destroy for multi-output op SSA.
    while (emitter.hasCudaEvent()) {
      os << "cudaEventDestroy(" << emitter.getEventName(emitter.popCudaEvent()) <<");\n";
    }

    os << "return;";
    return success();
}

LogicalResult printOperation(CudaEmitter &emitter, func::FuncOp funcOp) {
  CudaEmitter::Scope scope(emitter);
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

  INFO("add some fixed code")
  os << "int threads_per_block = 512;//fixed block setting for now\n";
  os << "\n";

  INFO("prepare some re-use var for ops")
  if (emitter.hasPplOp("onnx.Concat")) {
    os << "std::vector<ppl::common::TensorShape> pplConcatInputShapes;\n";
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
  if (hasPplOp("onnx.Abs")) { emitter.emitInclude(prefix.str().append("abs.h"), isLocal); }
  if (hasPplOp("onnx.Concat")) { emitter.emitInclude(prefix.str().append("memory/concat.h"), isLocal); }
  if (hasPplOp("onnx.MaxPoolSingleOut")) { emitter.emitInclude(prefix.str().append("nn/pooling_max.h"), isLocal);}
  os << "\n";
}

void printHelperFunction(CudaEmitter &emitter) {
  raw_indented_ostream &os = emitter.ostream();
  os << "__host__ int **createPPLDims(std::vector<ppl::common::TensorShape> &shapes) {\n";
  os << "  int **res = (int **)malloc(sizeof(*res) * shapes.size());\n";
  os << "  for (auto i = 0; i < shapes.size(); i++) {\n";
  os << "    int *dims = (int *)malloc(sizeof(*dims) *shapes.size());\n";
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
}

LogicalResult printOperation(CudaEmitter &emitter, ModuleOp moduleOp) {
  printFixedCode(emitter);
  emitter.collectPplInc(emitter, moduleOp);
  emitter.printPplInc(emitter);
  printHelperFunction(emitter);
  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

std::string CudaEmitter::getStreamName(Value value) {
  return  getOrCreateName(value.getDefiningOp()->getResult(0)).str() + "Stream";
}

std::string CudaEmitter::getEventName(Value value) {
  assert(value.getDefiningOp());
  return getOrCreateName(value.getDefiningOp()->getResult(0)).str() + "Event";
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
        if(failed(emitPPLShapeDeclaration(res))) {
          return op.emitError("failed to emit ppl shape declaration!");
        }
      }
    }

    //Name stream and event after res[0]. Declare and init stream and event for every ONNX op.
    INFO("create stream and event, naming after result[0].")
    Value res0 = op.getResult(0);
    os << "cudaStream_t " << getStreamName(res0) << ";\n";
    os << "cudaStreamCreate(&" << getStreamName(res0) << ");\n";
    os << "cudaEvent_t " << getEventName(res0) << ";\n";
    os << "cudaEventCreate(&" << getEventName(res0) << ");\n";
    pushCudaEvent(res0);

    INFO("Wait for events for every operand.")
    if (op.getNumOperands()) {
      for (auto operand : op.getOperands()) {
        if (NULL == operand.getDefiningOp()) { continue; }
        os << "cudaStreamWaitEvent(" << getStreamName(res0) << ", " << getEventName(operand) << ", 0);\n"; 
      }
    }
  }

  return success();
}

LogicalResult CudaEmitter::emitONNXPostOp(Operation &op) {
  INFO("Post proccess")
  INFO("1. emit event record")
  os << "cudaEventRecord(" << getEventName(op.getResult(0)) << ", " << getStreamName(op.getResult(0)) << ");\n";

  INFO("2. destroy first use operand`s stream (func args do not have definingOp and do not need streamdestroy)")
  INFO("3. free last use for every operands")
  if (op.getNumOperands()) {
    for (auto operand : op.getOperands()) {
      if (operand.getDefiningOp() && getValueFirstUse(operand) == (&op)) {
        os << "cudaStreamDestroy(" << getStreamName(operand) << ");\n";
      }
      if (getValueLastUse(operand) == (&op)) {
        if (failed(emitFree(operand))) {
          return failure();
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

  if (auto tType = value.getType().dyn_cast<TensorType>()) {
    os << "cudaMalloc((void**)&" << getOrCreateName(value) << ", "<< tType.getNumElements();
    os << " * sizeof(";
    if (failed(emitType(value.getLoc(), tType.getElementType()))) {
      return failure();
    }
    os << "))";
  }
  os << ";\n";

  return success();
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
                mlir::ONNXPowOp
                >(
              [&](auto op) {
                Operation *opop = op.getOperation();
                INFO(op.getLoc())
                if (failed(emitONNXPreOp(*opop)))         { return failure(); }
                if (failed(printOperation(*this, op)))    { return failure(); }
                if (failed(emitONNXPostOp(*opop)))        { return failure(); }
                return success();
          })
          // Func ops.
          .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
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


/// @brief 
/// @param input pre malloced device mem with input tensor data
/// @param inputShape LIST OF INTS. shape of input tensor
/// @param inputRank rank of input tensor
/// @param output pre malloced device mem
/// @param outputShape LIST OF INTS shape of output tensor
/// @param outputRank rank of output tensor
/// @param weight pre malloced device mem
/// @param weightShape LIST OF INTS shape of  weight tensor
/// @param weightRank rank of weight tensor
/// @param bias list of bias tensor
/// @param biasSize len(bias)
/// @param autoPadType auto pad type, default 0 is NOTSET
/// @param dilations LIST OF INTS dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
/// @param group (default 1)number of groups input channels and output channels are divided into
/// @param kernel_shape LIST OF INTS. number of groups input channels and output channels are divided into
/// @param kernel_rank rank of kernel shape
/// @param pads  LIST OF INTS. Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. 
/// @param pads_rank rank(pads)
/// @param strides LIST OF INTS. Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.
/// @param strides_rank rank(strides)
/// @return result code, 0 for success, error code o/w
int conv(
    float *input, int *inputShape, int inputRank,
    float *output, int *outputShape, int outputRank,
    float *weight, int *weightShape, int weightRank,
    float *bias, int biasSize,
    int autoPadType, 
    int *dilations, 
    int group, 
    int *kernel_shape, int kernel_rank,
    int *pads, int pads_rank,
    int *strides, int strides_rank
);

} // namespace mlir


