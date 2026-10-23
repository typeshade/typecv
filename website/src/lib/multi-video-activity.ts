import type { ViewGraphActivity } from '../../../src/reconstruction/view-graph.ts';

export interface VideoActivityLink {
  first: string;
  second: string;
  candidates: number;
  matched: number;
  matches: number;
  verified: number;
  accepted: number;
  inliers: number;
  active: boolean;
}

export interface MultiVideoActivity {
  stage: ViewGraphActivity['stage'];
  links: VideoActivityLink[];
  components: { index: number; videos: string[]; keyframes: number; reconstructed: boolean }[];
  limited: boolean;
  activeVideos: string[];
}

/** Bound display accounting independently of the graph/solver, including disconnected inputs. */
export const MAX_ACTIVITY_LINKS = 256;
export const MAX_ACTIVITY_THUMBNAILS = 64;
export const MAX_ACTIVITY_THUMBNAILS_PER_VIDEO = 4;

/** Aggregate frame pairs into video links. No descriptor, pixel or numerical result is retained. */
export class VideoActivityGraph {
  private readonly links = new Map<string, VideoActivityLink>();
  private components: MultiVideoActivity['components'] = [];
  private stage: MultiVideoActivity['stage'] = 'candidates';
  private limited = false;
  private activeVideos: string[] = [];

  update(event: ViewGraphActivity, videoOf: ReadonlyMap<number, string>): void {
    this.stage = event.stage;
    this.activeVideos = [];
    for (const link of this.links.values()) link.active = false;
    if (event.stage === 'components') {
      this.components = event.components.map((c) => ({
        index: c.index,
        videos: [...c.videoIds],
        keyframes: c.viewIds.length,
        reconstructed: c.reconstructed,
      }));
      return;
    }
    if (event.phase === 'start')
      this.activeVideos = [
        ...new Set(
          event.pairs
            .flatMap((pair) => [videoOf.get(pair.first), videoOf.get(pair.second)])
            .filter((id): id is string => id !== undefined),
        ),
      ];
    for (const pair of event.pairs) {
      const a = videoOf.get(pair.first),
        b = videoOf.get(pair.second);
      if (!a || !b || a === b) continue;
      const [first, second] = a < b ? [a, b] : [b, a];
      const key = first + ':' + second;
      let link = this.links.get(key);
      if (!link) {
        if (this.links.size >= MAX_ACTIVITY_LINKS) {
          this.limited = true;
          continue;
        }
        link = {
          first,
          second,
          candidates: 0,
          matched: 0,
          matches: 0,
          verified: 0,
          accepted: 0,
          inliers: 0,
          active: false,
        };
        this.links.set(key, link);
      }
      if (event.phase === 'start') {
        link.active = true;
        continue;
      }
      if (event.stage === 'candidates') link.candidates++;
      if (event.stage === 'matching') {
        link.matched++;
        link.matches += pair.matches ?? 0;
      }
      if (event.stage === 'verification') {
        link.verified++;
        if (pair.accepted) {
          link.accepted++;
          link.inliers += pair.inliers ?? 0;
        }
      }
    }
  }

  snapshot(): MultiVideoActivity {
    return {
      stage: this.stage,
      links: [...this.links.values()].map((link) => ({ ...link })),
      components: this.components.map((c) => ({ ...c, videos: [...c.videos] })),
      limited: this.limited,
      activeVideos: [...this.activeVideos],
    };
  }
}

export interface KeyframeThumbnail {
  videoId: string;
  frameId: number;
  time: number;
  image: string;
  features: number;
}

export function appendKeyframeThumbnail(
  previous: readonly KeyframeThumbnail[],
  next: KeyframeThumbnail,
): KeyframeThumbnail[] {
  const keep = previous
    .filter((frame) => frame.videoId !== next.videoId)
    .concat(
      previous
        .filter((frame) => frame.videoId === next.videoId && frame.frameId !== next.frameId)
        .slice(-(MAX_ACTIVITY_THUMBNAILS_PER_VIDEO - 1)),
      next,
    );
  return keep.slice(-MAX_ACTIVITY_THUMBNAILS);
}
