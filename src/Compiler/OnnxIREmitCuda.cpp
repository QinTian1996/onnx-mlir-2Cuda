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
#include "../Dialect/Mlir/DialectBuilder.hpp"



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
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
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
    CudaEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

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
};


CudaEmitter::CudaEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
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

static LogicalResult printCallOperation(CudaEmitter &emitter, Operation *callOp,
                                        StringRef callee) {

  raw_ostream &os = emitter.ostream();
  os << callee << "(";
  os << "TODO: args";
  os << ")";
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ModuleOp moduleOp) {
  CudaEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter,::mlir::ONNXAbsOp absOp) {
  //TODO:
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, ONNXLeakyReluOp leakyReluOp) {
  //TODO:
  return success();
}

LogicalResult printOperation(CudaEmitter &emitter, func::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();

  return printCallOperation(emitter, operation, callee);
}

LogicalResult printOperation(CudaEmitter &emitter, func::FuncOp funcOp) {
    //TODO:
    return success();

}

LogicalResult printOperation(CudaEmitter &emitter, func::ReturnOp returnOp) {
    //TODO:
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
          .Case<mlir::ONNXAbsOp, mlir::ONNXLeakyReluOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          //.Case<mlir::LiteralOp>([&](auto op) { return success(); })
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