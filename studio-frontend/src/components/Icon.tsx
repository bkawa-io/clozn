import type { SVGProps } from "react";

type IconName =
  | "observatory"
  | "runs"
  | "lens"
  | "compare"
  | "behavior"
  | "model"
  | "theme"
  | "inspector"
  | "investigation";

export function Icon({ name, ...props }: SVGProps<SVGSVGElement> & { name: IconName }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round" as const };
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" {...props}>
      {name === "observatory" && <g {...common}><circle cx="12" cy="12" r="3.2" /><path d="M3.5 12c2.4-4.6 5.2-6.9 8.5-6.9s6.1 2.3 8.5 6.9c-2.4 4.6-5.2 6.9-8.5 6.9S5.9 16.6 3.5 12Z" /><path d="M12 2.5v2.1M12 19.4v2.1" /></g>}
      {name === "runs" && <g {...common}><path d="M5 4.5h14v15H5z" /><path d="M8 8h8M8 12h8M8 16h5" /></g>}
      {name === "lens" && <g {...common}><circle cx="10.5" cy="10.5" r="5.5" /><path d="m14.6 14.6 5 5M8 10.5h5M10.5 8v5" /></g>}
      {name === "compare" && <g {...common}><path d="M4 5h6v14H4zM14 5h6v14h-6z" /><path d="m10 9 4 2M10 15l4-2" /></g>}
      {name === "behavior" && <g {...common}><path d="M4 7h9M17 7h3M4 17h3M11 17h9M13 4v6M8 14v6" /></g>}
      {name === "model" && <g {...common}><path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z" /><path d="m4 12 8 4.5 8-4.5M4 16.5 12 21l8-4.5" /></g>}
      {name === "theme" && <g {...common}><circle cx="12" cy="12" r="8" /><path d="M12 4a8 8 0 0 0 0 16V4Z" /></g>}
      {name === "inspector" && <g {...common}><path d="M4 5h16v14H4zM15 5v14M7.5 9h4M7.5 13h4" /></g>}
      {name === "investigation" && <g {...common}><path d="M5 5v14M5 8h4M5 13h3M12 5v6" /><circle cx="12" cy="14" r="1.4" /><path d="M12 15.4V19" /><path d="m15 8 4-2.5M19 5.5V8" /></g>}
    </svg>
  );
}
