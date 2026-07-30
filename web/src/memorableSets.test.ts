import { expect, test } from "vitest";

import { memorableStories } from "./memorableSets";

test("keeps every memorable chapter within the curated game length", () => {
  const chapters = memorableStories.flatMap((story) => story.chapters);
  const playIds = chapters.flatMap((chapter) =>
    chapter.scenarios.flatMap((scenario) =>
      scenario.pitches.map((pitch) => pitch.playId),
    ),
  );

  expect(memorableStories).toHaveLength(4);
  expect(chapters).toHaveLength(5);
  chapters.forEach((chapter) => {
    const pitchCount = chapter.scenarios.reduce(
      (sum, scenario) => sum + scenario.pitches.length,
      0,
    );
    if (chapter.id === "bal-bottom-9") {
      expect(chapter.scenarios).toHaveLength(6);
      expect(pitchCount).toBe(29);
      return;
    }
    expect(chapter.scenarios.length).toBeGreaterThanOrEqual(3);
    expect(chapter.scenarios.length).toBeLessThanOrEqual(4);
    expect(pitchCount).toBeGreaterThanOrEqual(12);
    expect(pitchCount).toBeLessThanOrEqual(18);
  });
  expect(new Set(playIds).size).toBe(92);
  chapters.forEach((chapter) =>
    chapter.scenarios.forEach((scenario) =>
      scenario.pitches.forEach((pitch) => {
        expect(pitch.modelPrediction?.modelVersion).toBe("V8.4");
        expect(scenario.availableGroups).toContain(
          pitch.modelPrediction?.selected,
        );
        expect(
          Object.values(pitch.modelPrediction?.probabilities ?? {}).reduce(
            (sum, probability) => sum + probability,
            0,
          ),
        ).toBeCloseTo(1, 5);
      }),
    ),
  );
});
