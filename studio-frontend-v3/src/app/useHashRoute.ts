import { useEffect, useState } from "react";
import { readRoute, type StudioRoute } from "./router";

export function useHashRoute(): StudioRoute {
  const [route, setRoute] = useState(() => readRoute());
  useEffect(() => {
    const update = () => setRoute(readRoute());
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return route;
}
