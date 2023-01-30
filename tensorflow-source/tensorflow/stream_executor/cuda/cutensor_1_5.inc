// Auto-generated, do not edit.

extern "C" {
cutensorStatus_t cutensorInit(cutensorHandle_t* handle) {
  using FuncPtr = cutensorStatus_t (*)(cutensorHandle_t*);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorInit");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cutensorStatus_t cutensorHandleDetachPlanCachelines(cutensorHandle_t* handle) {
  using FuncPtr = cutensorStatus_t (*)(cutensorHandle_t*);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cutensorHandleDetachPlanCachelines");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle);
}

cutensorStatus_t cutensorHandleAttachPlanCachelines(
    cutensorHandle_t* handle, cutensorPlanCacheline_t cachelines[],
    const uint32_t numCachelines) {
  using FuncPtr = cutensorStatus_t (*)(
      cutensorHandle_t*, cutensorPlanCacheline_t[], const uint32_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cutensorHandleAttachPlanCachelines");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, cachelines, numCachelines);
}

cutensorStatus_t cutensorHandleWriteCacheToFile(const cutensorHandle_t* handle,
                                                const char filename[]) {
  using FuncPtr = cutensorStatus_t (*)(const cutensorHandle_t*, const char[]);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorHandleWriteCacheToFile");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, filename);
}

cutensorStatus_t cutensorHandleReadCacheFromFile(cutensorHandle_t* handle,
                                                 const char filename[],
                                                 uint32_t* numCachelinesRead) {
  using FuncPtr =
      cutensorStatus_t (*)(const cutensorHandle_t*, const char[], uint32_t*);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorHandleReadCacheFromFile");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, filename, numCachelinesRead);
}

cutensorStatus_t cutensorInitTensorDescriptor(
    const cutensorHandle_t* handle, cutensorTensorDescriptor_t* desc,
    const uint32_t numModes, const int64_t extent[], const int64_t stride[],
    cudaDataType_t dataType, cutensorOperator_t unaryOp) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, cutensorTensorDescriptor_t*, const uint32_t,
      const int64_t[], const int64_t[], cudaDataType_t, cutensorOperator_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorInitTensorDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, desc, numModes, extent, stride, dataType, unaryOp);
}

cutensorStatus_t cutensorElementwiseTrinary(
    const cutensorHandle_t* handle, const void* alpha, const void* A,
    const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
    const void* beta, const void* B, const cutensorTensorDescriptor_t* descB,
    const int32_t modeB[], const void* gamma, const void* C,
    const cutensorTensorDescriptor_t* descC, const int32_t modeC[], void* D,
    const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
    cutensorOperator_t opAB, cutensorOperator_t opABC,
    cudaDataType_t typeScalar, const cudaStream_t stream) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const void*, const void*,
      const cutensorTensorDescriptor_t*, const int32_t[], const void*,
      const void*, const cutensorTensorDescriptor_t*, const int32_t[],
      const void*, const void*, const cutensorTensorDescriptor_t*,
      const int32_t[], void*, const cutensorTensorDescriptor_t*,
      const int32_t[], cutensorOperator_t, cutensorOperator_t, cudaDataType_t,
      const cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorElementwiseTrinary");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma,
                  C, descC, modeC, D, descD, modeD, opAB, opABC, typeScalar,
                  stream);
}

cutensorStatus_t cutensorElementwiseBinary(
    const cutensorHandle_t* handle, const void* alpha, const void* A,
    const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
    const void* gamma, const void* C, const cutensorTensorDescriptor_t* descC,
    const int32_t modeC[], void* D, const cutensorTensorDescriptor_t* descD,
    const int32_t modeD[], cutensorOperator_t opAC, cudaDataType_t typeScalar,
    cudaStream_t stream) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const void*, const void*,
      const cutensorTensorDescriptor_t*, const int32_t[], const void*,
      const void*, const cutensorTensorDescriptor_t*, const int32_t[], void*,
      const cutensorTensorDescriptor_t*, const int32_t[], cutensorOperator_t,
      cudaDataType_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorElementwiseBinary");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D,
                  descD, modeD, opAC, typeScalar, stream);
}

cutensorStatus_t cutensorPermutation(
    const cutensorHandle_t* handle, const void* alpha, const void* A,
    const cutensorTensorDescriptor_t* descA, const int32_t modeA[], void* B,
    const cutensorTensorDescriptor_t* descB, const int32_t modeB[],
    const cudaDataType_t typeScalar, const cudaStream_t stream) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const void*, const void*,
      const cutensorTensorDescriptor_t*, const int32_t[], void*,
      const cutensorTensorDescriptor_t*, const int32_t[], const cudaDataType_t,
      const cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorPermutation");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, A, descA, modeA, B, descB, modeB, typeScalar,
                  stream);
}

cutensorStatus_t cutensorInitContractionDescriptor(
    const cutensorHandle_t* handle, cutensorContractionDescriptor_t* desc,
    const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
    const uint32_t alignmentRequirementA,
    const cutensorTensorDescriptor_t* descB, const int32_t modeB[],
    const uint32_t alignmentRequirementB,
    const cutensorTensorDescriptor_t* descC, const int32_t modeC[],
    const uint32_t alignmentRequirementC,
    const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
    const uint32_t alignmentRequirementD, cutensorComputeType_t typeCompute) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, cutensorContractionDescriptor_t*,
      const cutensorTensorDescriptor_t*, const int32_t[], const uint32_t,
      const cutensorTensorDescriptor_t*, const int32_t[], const uint32_t,
      const cutensorTensorDescriptor_t*, const int32_t[], const uint32_t,
      const cutensorTensorDescriptor_t*, const int32_t[], const uint32_t,
      cutensorComputeType_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cutensorInitContractionDescriptor");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, desc, descA, modeA, alignmentRequirementA, descB,
                  modeB, alignmentRequirementB, descC, modeC,
                  alignmentRequirementC, descD, modeD, alignmentRequirementD,
                  typeCompute);
}

cutensorStatus_t cutensorContractionDescriptorSetAttribute(
    const cutensorHandle_t* handle, cutensorContractionDescriptor_t* desc,
    cutensorContractionDescriptorAttributes_t attr, const void* buf,
    size_t sizeInBytes) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, cutensorContractionDescriptor_t*,
      cutensorContractionDescriptorAttributes_t, const void*, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cutensorContractionDescriptorSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, desc, attr, buf, sizeInBytes);
}

cutensorStatus_t cutensorInitContractionFind(const cutensorHandle_t* handle,
                                             cutensorContractionFind_t* find,
                                             const cutensorAlgo_t algo) {
  using FuncPtr =
      cutensorStatus_t (*)(const cutensorHandle_t*, cutensorContractionFind_t*,
                           const cutensorAlgo_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorInitContractionFind");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, find, algo);
}

cutensorStatus_t cutensorContractionFindSetAttribute(
    const cutensorHandle_t* handle, cutensorContractionFind_t* find,
    cutensorContractionFindAttributes_t attr, const void* buf,
    size_t sizeInBytes) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, cutensorContractionFind_t*,
      cutensorContractionFindAttributes_t, const void*, size_t);
  static auto func_ptr =
      LoadSymbol<FuncPtr>("cutensorContractionFindSetAttribute");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, find, attr, buf, sizeInBytes);
}

cutensorStatus_t cutensorContractionGetWorkspaceSize(
    const cutensorHandle_t* handle, const cutensorContractionDescriptor_t* desc,
    const cutensorContractionFind_t* find,
    const cutensorWorksizePreference_t pref, uint64_t* workspaceSize) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const cutensorContractionDescriptor_t*,
      const cutensorContractionFind_t*, const cutensorWorksizePreference_t,
      uint64_t*);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorContractionGetWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, desc, find, pref, workspaceSize);
}

cutensorStatus_t cutensorInitContractionPlan(
    const cutensorHandle_t* handle, cutensorContractionPlan_t* plan,
    const cutensorContractionDescriptor_t* desc,
    const cutensorContractionFind_t* find, const uint64_t workspaceSize) {
  using FuncPtr =
      cutensorStatus_t (*)(const cutensorHandle_t*, cutensorContractionPlan_t*,
                           const cutensorContractionDescriptor_t*,
                           const cutensorContractionFind_t*, const uint64_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorInitContractionPlan");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, plan, desc, find, workspaceSize);
}

cutensorStatus_t cutensorContraction(const cutensorHandle_t* handle,
                                     const cutensorContractionPlan_t* plan,
                                     const void* alpha, const void* A,
                                     const void* B, const void* beta,
                                     const void* C, void* D, void* workspace,
                                     uint64_t workspaceSize,
                                     cudaStream_t stream) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const cutensorContractionPlan_t*, const void*,
      const void*, const void*, const void*, const void*, void*, void*,
      uint64_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorContraction");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, plan, alpha, A, B, beta, C, D, workspace,
                  workspaceSize, stream);
}

cutensorStatus_t cutensorContractionMaxAlgos(int32_t* maxNumAlgos) {
  using FuncPtr = cutensorStatus_t (*)(int32_t*);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorContractionMaxAlgos");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(maxNumAlgos);
}

cutensorStatus_t cutensorReduction(
    const cutensorHandle_t* handle, const void* alpha, const void* A,
    const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
    const void* beta, const void* C, const cutensorTensorDescriptor_t* descC,
    const int32_t modeC[], void* D, const cutensorTensorDescriptor_t* descD,
    const int32_t modeD[], cutensorOperator_t opReduce,
    cutensorComputeType_t typeCompute, void* workspace, uint64_t workspaceSize,
    cudaStream_t stream) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const void*, const void*,
      const cutensorTensorDescriptor_t*, const int32_t[], const void*,
      const void*, const cutensorTensorDescriptor_t*, const int32_t[], void* D,
      const cutensorTensorDescriptor_t*, const int32_t[], cutensorOperator_t,
      cutensorComputeType_t, void*, uint64_t, cudaStream_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorReduction");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, alpha, A, descA, modeA, beta, C, descC, modeC, D,
                  descD, modeD, opReduce, typeCompute, workspace, workspaceSize,
                  stream);
}

cutensorStatus_t cutensorReductionGetWorkspaceSize(
    const cutensorHandle_t* handle, const void* A,
    const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
    const void* C, const cutensorTensorDescriptor_t* descC,
    const int32_t modeC[], const void* D,
    const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
    cutensorOperator_t opReduce, cutensorComputeType_t typeCompute,
    uint64_t* workspaceSize) {
  using FuncPtr = cutensorStatus_t (*)(
      const cutensorHandle_t*, const void*, const cutensorTensorDescriptor_t*,
      const int32_t[], const void*, const cutensorTensorDescriptor_t*,
      const int32_t[], const void*, const cutensorTensorDescriptor_t*,
      const int32_t[], cutensorOperator_t, cutensorComputeType_t, uint64_t*);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorReductionGetWorkspaceSize");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, A, descA, modeA, C, descC, modeC, D, descD, modeD,
                  opReduce, typeCompute, workspaceSize);
}

cutensorStatus_t cutensorGetAlignmentRequirement(
    const cutensorHandle_t* handle, const void* ptr,
    const cutensorTensorDescriptor_t* desc, uint32_t* alignmentRequirement) {
  using FuncPtr =
      cutensorStatus_t (*)(const cutensorHandle_t*, const void*,
                           const cutensorTensorDescriptor_t*, uint32_t*);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorGetAlignmentRequirement");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr(handle, ptr, desc, alignmentRequirement);
}

const char* cutensorGetErrorString(const cutensorStatus_t error) {
  using FuncPtr = const char* (*)(const cutensorStatus_t);
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorGetErrorString");
  if (!func_ptr) return "no get";  // GetSymbolNotFoundError();
  return func_ptr(error);
}

size_t cutensorGetVersion() {
  using FuncPtr = size_t (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorGetVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

size_t cutensorGetCudartVersion() {
  using FuncPtr = size_t (*)();
  static auto func_ptr = LoadSymbol<FuncPtr>("cutensorGetCudartVersion");
  if (!func_ptr) return GetSymbolNotFoundError();
  return func_ptr();
}

}  // extern "C"
