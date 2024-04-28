#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/IndentedOstream.h"


namespace onnx_mlir {

mlir::LogicalResult translateToCuda(mlir::Operation *op, llvm::raw_ostream &os,
                                    bool declareVariablesAtTop);

}

namespace mlir {

void registerToCudaTranslation();

}