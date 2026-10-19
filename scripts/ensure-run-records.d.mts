/** Declarations for the deploy-time script, so that its unit test typechecks. */
export declare const NAMESPACE: string;
export declare const PLACEHOLDER: string;
/** The id of the `typecv-records` namespace in `wrangler kv namespace list` output, if present. */
export declare function findNamespaceId(listOutput: string, worker?: string): string | undefined;
/** The id printed by `wrangler kv namespace create`, if present. */
export declare function createdNamespaceId(createOutput: string): string | undefined;
/** The configuration text with the namespace bound to `id`, or unbound when `id` is undefined. */
export declare function bindNamespace(config: string, id: string | undefined): string;
