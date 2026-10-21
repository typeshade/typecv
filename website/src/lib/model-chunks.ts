/**
 * The final map of a component crosses from the worker to the page as many small messages, not
 * one. The page deserializes every message on its main thread. A final map of 50,000 landmarks
 * with their observations took 236 ms as one message (measured 2026-10-21), a long task, and a
 * part of 4,000 landmarks takes about a tenth of that.
 */
import type { Scene, ScenePoint } from './sfm-scene';

/** The landmarks in one part. */
export const MODEL_PART_LANDMARKS = 4000;

export interface ModelPart {
  type: 'modelPart';
  component: number;
  /** The views come with the first part only. */
  views?: Scene['views'];
  landmarks: ScenePoint[];
}

/**
 * The messages for a final `model` message: the parts, then the `model` message without its
 * reconstruction and with the number of parts. A model that fits one part goes unchanged.
 */
export function splitModel<T extends { type: string; component: number; reconstruction?: Scene }>(
  message: T,
  size = MODEL_PART_LANDMARKS,
): (T | ModelPart)[] {
  const scene = message.reconstruction;
  if (message.type !== 'model' || !scene?.landmarks || scene.landmarks.length <= size)
    return [message];
  const out: (T | ModelPart)[] = [];
  for (let at = 0; at < scene.landmarks.length; at += size)
    out.push({
      type: 'modelPart',
      component: message.component,
      ...(at === 0 ? { views: scene.views } : {}),
      landmarks: scene.landmarks.slice(at, at + size),
    });
  out.push({ ...message, reconstruction: undefined, views: scene.views, parts: out.length });
  return out;
}

/** The page side: collects the parts of each component and gives back the whole message. */
export class ModelAssembler {
  private readonly parts = new Map<number, ScenePoint[][]>();

  /** The message to handle, or undefined for a part (kept until its `model` message arrives). */
  accept<M extends { type: string; component?: number; parts?: number; views?: Scene['views'] }>(
    message: M,
  ): M | undefined {
    if (message.type === 'modelPart') {
      const part = message as unknown as ModelPart;
      const list = this.parts.get(part.component) ?? [];
      list.push(part.landmarks);
      this.parts.set(part.component, list);
      return undefined;
    }
    if (message.type === 'model' && message.parts !== undefined) {
      const list = this.parts.get(message.component!) ?? [];
      this.parts.delete(message.component!);
      return {
        ...message,
        reconstruction: { views: message.views ?? [], landmarks: list.flat() },
      };
    }
    return message;
  }
}
