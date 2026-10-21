/**
 * What a worker sends the page of the WebGPU adapter, for the run record (`gpu` in
 * `run-record.ts`): the identity strings, the two buffer limits, the compute limits that bound a
 * kernel's workgroup, and the feature names. The multi-video job and the SfM worker both use it.
 */
export interface AdapterFacts {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  fallbackAdapter: boolean;
  limits: {
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
    maxComputeWorkgroupStorageSize: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxStorageBuffersPerShaderStage: number;
  };
  /** Feature names, sorted: `shader-f16`, `subgroups`, `timestamp-query`, ... */
  features: string[];
}

/** The facts of a `GPUAdapter`. A limit the adapter does not report is 0. */
export function adapterFacts(requested: any): AdapterFacts {
  const info = requested.info ?? {};
  const limit = (name: string): number => {
    const value = requested.limits?.[name];
    return typeof value === 'number' && Number.isFinite(value) ? value : 0;
  };
  let features: string[] = [];
  try {
    features = [...(requested.features ?? [])].map(String).sort();
  } catch {
    features = [];
  }
  return {
    vendor: info.vendor ?? '',
    architecture: info.architecture ?? '',
    device: info.device ?? '',
    description: info.description ?? '',
    fallbackAdapter: requested.isFallbackAdapter === true,
    limits: {
      maxStorageBufferBindingSize: limit('maxStorageBufferBindingSize'),
      maxBufferSize: limit('maxBufferSize'),
      maxComputeWorkgroupStorageSize: limit('maxComputeWorkgroupStorageSize'),
      maxComputeInvocationsPerWorkgroup: limit('maxComputeInvocationsPerWorkgroup'),
      maxStorageBuffersPerShaderStage: limit('maxStorageBuffersPerShaderStage'),
    },
    features,
  };
}
