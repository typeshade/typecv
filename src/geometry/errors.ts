/** A valid estimation request had no sufficiently reliable geometric solution. */
export class GeometryEstimationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GeometryEstimationError';
  }
}
