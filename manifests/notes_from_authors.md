# Communications with dataset authors

## mpii: 3-person vs 4-person sessions + audio missing on empty seat

**From**: Huajian Qiu (PhD Student, Collaborative AI / VIS, University of
Stuttgart) — huajian.qiu@vis.uni-stuttgart.de

**Date**: response to 2026-06-08 query

**Context**: We observed that mpii test `001/subjectPos4` has visual
features (`openface2`, `openpose`, `videomae`, etc.) but no audio
features (`w2vbert2`, `egemapsv2`, `xlmr`), unlike groups 002-006.

**Author's reply**:

> A quick answer is that in the test set, the group 001 has only 3
> participants so there is no person at the position of SubjectPos04. In
> the dataset, there are some groups with 3 persons, while others have 4
> persons. The reason why there are precomputed visual features for
> SubjectPos04 of test group 001 while no audio features is unknown for
> me.

Excerpt from the original MPIIGroupInteraction README that explains why:

> The dataset consists of audio and video recordings of group discussions.
> Each discussion lasts for approximately 20 minutes and contains four
> videos, each displaying a frontal view of a single participant (for
> three-person interactions the video of the missing participant shows
> an empty seat). Audio recordings are provided by a single microphone
> positioned in the room.

> Seating arrangement: Participants are sitting at four discrete spots on
> an imaginary circle, with two participants facing each other. The
> numbering of seating positions is in counter-clockwise order:
>
>     3   4
>      2 1
>
> The video of ID X contains a frontal view of participant X. For
> example, video 2 is obtained from a camera positioned behind participant
> 4, directed towards participant 2.
>
> In addition to video recordings, microphones were positioned in front
> and slightly above of participants. The microphone from which the
> recorded audio is obtained for a given discussion is chosen randomly
> amongst these four microphone.

**Implications for our pipeline**:

1. `(session, role)` slots whose source dir contains no `{role}.*.stream`
   files should be skipped at build time — not zero-filled. This is the
   "empty seat" case.

2. Visual features may exist for an empty seat (just records the empty
   chair). They carry no informative signal but contribute no NaN either,
   so it's safe to keep them; downstream model sees label_mask=False
   and ignores during loss.

3. Audio features are missing per-position for empty seats because audio
   is from a single shared mic that's randomly attributed to participants.
   The 3-of-4 audio files are conceptually all the SAME audio — see if
   this matters for fusion.

Per-session 3-vs-4 inventory:

```
mpii val:
  4-person: 008, 009, 010, 028
  3-person: 026 (Pos1 empty), 027 (Pos1 empty)

mpii test:
  4-person: 002, 003, 004, 005, 006
  3-person: 001 (Pos4 empty)
```
