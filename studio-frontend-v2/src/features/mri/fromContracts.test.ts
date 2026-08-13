import { expect, it } from "vitest";
import { projectRecordedMriSpecimen } from "./fromContracts";

it("never manufactures internal measurements from retained response tokens", () => {
  const specimen = projectRecordedMriSpecimen({ id: "run-a", responseTokens: ["A", " token"] });
  expect(specimen.tokens).toHaveLength(2);
  expect(specimen.layers).toEqual([]);
  expect(specimen.channels.every((channel) => channel.capability === "not-reported")).toBe(true);
  expect(specimen.observationsByChannelId).toEqual({});
});
