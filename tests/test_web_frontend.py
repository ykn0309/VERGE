"""Execute pure dashboard presentation logic without a browser dependency."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_timeline_folds_origin_add_without_changing_credit_events():
    source = (Path(__file__).parents[1] / "verge/web_assets/app.js").read_text()
    function = source.split("function experienceTimeline(e) {", 1)[1].split("\nfunction lineChart", 1)[0]
    subprocess.run(["node", "-e", "function experienceTimeline(e) {" + function + """
const assert = require('node:assert/strict');
const prior = {kind:'prior', origin_kind:'add', sample_id:'a', alpha:1, beta:1, confidence:.5};
const add = {kind:'fusion', fusion:'ADD', sample_id:'a', candidate_id:'c', alpha:1, beta:1, confidence:.5};
const credit = {kind:'downstream_credit', sample_id:'b', alpha:1.2, beta:1, confidence:1.2/2.2};
const merge = {kind:'fusion', fusion:'MERGE', sample_id:'c'};
const original = [prior, add, credit, merge];
const events = experienceTimeline({evolution:original});
assert.equal(events.length, 3);
assert.equal(events[0].candidate_id, 'c');
assert.equal(events[0].alpha, 1);
assert.equal(events[1], credit);
assert.equal(events[2], merge);
assert.equal(original.length, 4);
assert.equal(original[0].kind, 'prior');
const creditedAdd = {...add, kind:'fusion_credit', credit:.1, alpha:1.1, confidence:1.1/2.1};
const creditedEvents = experienceTimeline({evolution:[prior, creditedAdd, credit]});
assert.equal(creditedEvents.length, 2);
assert.equal(creditedEvents[0].credit, .1);
assert.equal(creditedEvents[0].confidence, 1.1/2.1);
// Never fold missing/unmatched origin or MERGE into the prior.
for (const sequence of [[prior], [prior, credit], [prior, merge],
    [prior, {...add, sample_id:'other'}, credit], [{...prior, origin_kind:'source'}, add]]) {
  assert.equal(experienceTimeline({evolution:sequence}), sequence);
}
assert.deepEqual(experienceTimeline({evolution:[]}), []);
"""], check=True, capture_output=True, text=True)
