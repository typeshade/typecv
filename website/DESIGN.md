# TypeCV website design

The public website follows the TypeNN design system in `references/typenn/DESIGN.md` (source repository revision `61259c3733148a8665466163ae1c5c2b920b129c`). The implementation uses Astro, React islands, Ant Design 5, Tailwind CSS 4, IBM Plex fonts, a responsive documentation shell, and persisted light/dark themes.

TypeCV-specific content gives precedence to runnable computer vision operations, linked technical documentation, and reproducible experiment records. The geometry playground reports real result and execution metrics. The video lab labels the currently available frame preview separately from the planned GPU reconstruction pipeline. It does not display a recorded point cloud as if it came from the selected video.

The site is public and does not request sign-in. Video decoding stays in the browser. Controls use standard labels, visible keyboard focus, and mobile layouts. Colors and typography inherit the TypeNN tokens; illustrations and data displays must communicate actual algorithm output rather than decorative accuracy claims.
