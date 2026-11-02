import { attitudeTone, humanize } from "@/lib/format";
import type { Civilization, World } from "@/lib/types";

export function WorldView({ world }: { world: World }) {
  const focal = world.civilizations.find((c) => c.id === world.focal_civilization_id)!;
  const neighbors = world.civilizations.filter((c) => c.id !== focal.id);

  return (
    <div className="world">
      <section className="card focal">
        <p className="eyebrow">Your civilization</p>
        <h2>{focal.name}</h2>
        <CivilizationFacts civ={focal} />
        {focal.description && <p className="description">{focal.description}</p>}
        {focal.generated_fields.length > 0 && (
          <p className="hint">
            <span className="ai-tag">AI</span> marks characteristics chosen for you.
          </p>
        )}
      </section>

      <h3 className="section-title">Neighbors</h3>
      <div className="neighbors">
        {neighbors.map((n) => {
          const r = world.relationships.find(
            (rel) => rel.civilization_a_id === focal.id && rel.civilization_b_id === n.id,
          );
          const sketch = world.language_sketches.find((s) => s.civilization_id === n.id);
          return (
            <section key={n.id} className="card">
              <h4>{n.name}</h4>
              <CivilizationFacts civ={n} showGenerated={false} />
              {r && (
                <div className="relationship">
                  <AttitudeLine from={focal.name} to={n.name} attitude={r.attitude_a_to_b} />
                  <AttitudeLine from={n.name} to={focal.name} attitude={r.attitude_b_to_a} />
                  <dl className="facts compact">
                    <dt>Power</dt>
                    <dd>{humanize(r.power_of_b_relative_to_a)} than {focal.name}</dd>
                    <dt>Separated by</dt>
                    <dd>{humanize(r.separation)}</dd>
                    <dt>Contact</dt>
                    <dd>
                      {humanize(r.contact_intensity)}: {r.contact_types.map(humanize).join(", ")}
                    </dd>
                  </dl>
                </div>
              )}
              {sketch && (
                <p className="sounds">
                  Sounds: <code>{sketch.phonology.phonemes.map((p) => p.grapheme).join(" ")}</code>
                </p>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}

function CivilizationFacts({ civ, showGenerated = true }: { civ: Civilization; showGenerated?: boolean }) {
  const ai = (field: string) =>
    showGenerated && civ.generated_fields.includes(field) ? <span className="ai-tag">AI</span> : null;
  return (
    <dl className="facts">
      <dt>Land</dt>
      <dd>
        {humanize(civ.geography)} {ai("geography")}
      </dd>
      <dt>Economy</dt>
      <dd>
        {humanize(civ.economy)} {ai("economy")}
      </dd>
      {civ.social_organization && (
        <>
          <dt>Society</dt>
          <dd>
            {humanize(civ.social_organization)} {ai("social_organization")}
          </dd>
        </>
      )}
      <dt>Culture</dt>
      <dd>
        {civ.cultural_traits.length ? civ.cultural_traits.map(humanize).join(", ") : "—"} {ai("cultural_traits")}
      </dd>
    </dl>
  );
}

function AttitudeLine({ from, to, attitude }: { from: string; to: string; attitude: string }) {
  return (
    <p className="attitude">
      <span>
        {from} → {to}
      </span>
      <span className={`badge ${attitudeTone(attitude)}`}>{humanize(attitude)}</span>
    </p>
  );
}
