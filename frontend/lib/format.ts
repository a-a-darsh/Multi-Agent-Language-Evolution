/** "pastoral_nomadic" -> "Pastoral nomadic" */
export function humanize(value: string): string {
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

const ATTITUDE_TONE: Record<string, "good" | "neutral" | "bad"> = {
  admiration: "good",
  respect: "good",
  indifference: "neutral",
  distrust: "bad",
  contempt: "bad",
  hostility: "bad",
};

export function attitudeTone(attitude: string): "good" | "neutral" | "bad" {
  return ATTITUDE_TONE[attitude] ?? "neutral";
}
