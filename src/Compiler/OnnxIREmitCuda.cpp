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

#include <utility>
#include <stack>
#include <set>
#include "../Dialect/Mlir/DialectBuilder.hpp"
#include <llvm/IR/DerivedTypes.h>



#define DEBUG_TYPE "translate-to-cuda"

using namespace mlir;
using llvm::formatv;


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

  return emitError(loc, "cannot emit unkown type ") << type;
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

  if ((!tTypeX) || (!tTypeY)) {
    return leakyReluOp.emitOpError("operand not valid tensor type!");
  }

  if (tTypeX.getNumElements() != tTypeY.getNumElements()) {
    return leakyReluOp.emitOpError("operand size not match!");
  }
  os << "onnxLeakyReLU";
  os << "<<<(" << tTypeX.getNumElements() << " + threads_per_block - 1)/threads_per_block, threads_per_block>>>";
  os << "(";
  os << emitter.getOrCreateName(x); //X
  os << ", ";
  os << emitter.getOrCreateName(y); //Y
  os << ", ";
  os << tTypeX.getNumElements(); //size
  os << ", ";
  os << (tTypeX.getElementTypeBitWidth() >> 3); //stride
  os << ", ";
  os << alpha.convertToFloat(); //alpha
  os << ");";
  return success();
}

template <typename T>
LogicalResult printONNXArithmetic(CudaEmitter &emitter, T arithmeticOp) {
  raw_indented_ostream &os = emitter.ostream();
  Value a = arithmeticOp.getA();
  Value b = arithmeticOp.getB();
  Value c = arithmeticOp.getC();
  auto tTypeA = a.getType().dyn_cast_or_null<TensorType>();
  auto tTypeB = b.getType().dyn_cast_or_null<TensorType>();
  auto tTypeC = c.getType().dyn_cast_or_null<TensorType>();
  auto shapeA = tTypeA.getShape();
  auto shapeB = tTypeB.getShape();
  auto shapeC = tTypeC.getShape();

  //Not support broadcast for diff shape for now.
  if (!shapeA.equals(shapeB) || !shapeA.equals(shapeC)) {
    return arithmeticOp.emitOpError("operand size not match!");
  }
  //Not support dynamic shape for now.
  if (tTypeA.getNumDynamicDims()) {
    return arithmeticOp.emitOpError("Dynamic shape not supported!");
  }

  auto width = shapeA[0];
  auto height = tTypeA.getNumElements() / width;;
  auto channel = 1;
  auto stream = emitter.getStreamName(c);
  /*
  *   c = a + b
  *   ppl::cv::cuda::Add<float, 3>( stream, height, width,
  *                                 channel * width, a,
  *                                 channel * width, b,
  *                                 channel * width, c);
  *
  */
  os << "ppl::cv::cuda::";
  if      (dyn_cast_or_null<mlir::ONNXAddOp>(&arithmeticOp)) { os << "Add";      }
  else if (dyn_cast_or_null<mlir::ONNXSubOp>(&arithmeticOp)) { os << "Subtract"; }
  else if (dyn_cast_or_null<mlir::ONNXMulOp>(&arithmeticOp)) { os << "Mul";      }
  else if (dyn_cast_or_null<mlir::ONNXDivOp>(&arithmeticOp)) { os << "Div";      }
  else { return arithmeticOp.emitError("op is not onnx arithmetic!");   }

  os << "<";
  if (failed(emitter.emitType(a.getLoc(), tTypeA.getElementType()))) {
    return failure();
  }
  os << ", " << channel << ">(";
  os << stream                      << ", "; // stream
  os << height                      << ", "; // height
  os << width                       << ", "; // width
  os << channel * width             << ", "; // stride of a
  os << emitter.getOrCreateName(a)  << ", "; // var name of a
  if (!dyn_cast_or_null<mlir::ONNXSubOp>(&arithmeticOp)) {
    os << channel * width             << ", "; // stride of b
  }
  os << emitter.getOrCreateName(b)  << ", "; // var name of b
  os << channel * width             << ", "; // stride of c
  os << emitter.getOrCreateName(c);          // var name of c
  os << ");\n";

  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXAddOp addOp) {
  return printONNXArithmetic<ONNXAddOp>(emitter, addOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXMulOp mulOp) {
  return printONNXArithmetic<ONNXMulOp>(emitter, mulOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXSubOp subOp) {
  return printONNXArithmetic<ONNXSubOp>(emitter, subOp);
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXDivOp divOp) {
  return printONNXArithmetic<ONNXDivOp>(emitter, divOp);
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

  //add some fiexed code
  os << "int threads_per_block = 512; //fixed block setting for now\n";


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
  StringRef prefix = "ppl/cv/cuda/";
  if (hasPplOp("onnx.Add")) { emitter.emitInclude(prefix.str().append("arithmetic.h"), false); }
  if (hasPplOp("onnx.Abs")) { emitter.emitInclude(prefix.str().append("abs.h"), false); }
  //TODO: add other ops

  os << "\n";
}

LogicalResult printOperation(CudaEmitter &emitter, ModuleOp moduleOp) {
  printFixedCode(emitter);
  emitter.collectPplInc(emitter, moduleOp);
  emitter.printPplInc(emitter);
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
    //SSA  var declaration
    for (auto res : op.getResults()) {
      if (failed(emitDeclaration(res))) {
        return op.emitOpError("unable to declare ssa var.");
      }
    }

    //Name stream and event after res[0]. Declare and init stream and event for every ONNX op.
    Value res0 = op.getResult(0);
    os << "cudaStream_t " << getStreamName(res0) << ";\n";
    os << "cudaStreamCreate(&" << getStreamName(res0) << ");\n";
    os << "cudaEvent_t " << getEventName(res0) << ";\n";
    os << "cudaEventCreate(&" << getEventName(res0) << ");\n";
    pushCudaEvent(res0);

    //Wait for events for every operand
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
  // 1. emit event record
  os << "cudaEventRecord(" << getEventName(op.getResult(0)) << ", " << getStreamName(op.getResult(0)) << ");\n";

  if (op.getNumOperands()) {
    for (auto operand : op.getOperands()) {
      // 2. destroy first use operand`s stream (func args do not have definingOp and do not need streamdestroy)
      if (operand.getDefiningOp() && getValueFirstUse(operand) == (&op)) {
        os << "cudaStreamDestroy(" << getStreamName(operand) << ");\n";
      }

      // 3. free last use for every operands
      if (getValueLastUse(operand) == (&op)) {
        if (failed(emitFree(operand))) {
          return failure();
        }
      }
    }
  }

  return success();
}

LogicalResult CudaEmitter::emitDeclaration(Value &value) {
  if (failed(this->emitType(value.getLoc(), value.getType()))) {
    return failure();
  }

  os << getOrCreateName(value) << ";\n";

  if (auto tType = value.getType().dyn_cast<TensorType>()) {
    os << "cudaMalloc((void**)&" << getOrCreateName(value) << ", "<< tType.getNumElements() << " * sizeof(";
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
                mlir::ONNXAddOp, mlir::ONNXMulOp, mlir::ONNXDivOp, mlir::ONNXSubOp
                >(
              [&](auto op) {
                Operation *opop = op.getOperation();
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
            return op.emitOpError("unable to find printer for op");
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

} // namespace mlir