# Findings, severity and planning disposition

[Planning draft](README.md) · [Work items](WORK_ITEMS.md) · [Tracking data](tracking.json)

This is a planning overlay on the unchanged validated audit. Severity means impact **when the stated route/input is exposed**; it is not incident probability. Original P1–P3 priorities and confidence are retained. Memory/resource failures can affect availability at scale without a reproduced outage. High does not mean every caller is affected. Qualified or unverified records do not become confirmed bugs by receiving an owner.

Each of the 131 records has one primary work-item owner. Related records share tasks; context, investigations and rejected inferences are not additional bugs. Source evidence and full acceptance criteria remain in the audit and work-item catalog. All statuses are planning dispositions, not fixes.

| Record | Impact | Severity | Audit priority / confidence | Primary owner | Disposition |
|---|---|---|---|---|---|
| [API-01](../../../audits/2026-09-14/CLAIMS.md#api-01) | Feature / result correctness | High when exposed | P1 / confirmed-reproduced | [R01](WORK_ITEMS.md#r01) | Plan regression and repair |
| [API-02](../../../audits/2026-09-14/CLAIMS.md#api-02) | Contract / integration | Medium | P2 / confirmed-reproduced | [E06](WORK_ITEMS.md#e06) | Plan regression and repair |
| [API-03](../../../audits/2026-09-14/CLAIMS.md#api-03) | Contract / integration | Medium | P2 / confirmed-reproduced | [E03](WORK_ITEMS.md#e03) | Plan regression and repair |
| [API-04](../../../audits/2026-09-14/CLAIMS.md#api-04) | Resource / availability | Medium | P2 / confirmed-reproduced | [E02](WORK_ITEMS.md#e02) | Plan regression and repair |
| [API-05](../../../audits/2026-09-14/CLAIMS.md#api-05) | Resource / availability | Medium | P2 / confirmed-reproduced | [E04](WORK_ITEMS.md#e04) | Plan regression and repair |
| [API-06](../../../audits/2026-09-14/CLAIMS.md#api-06) | Resource / availability | Medium | P2 / confirmed-reproduced | [E04](WORK_ITEMS.md#e04) | Plan regression and repair |
| [API-07](../../../audits/2026-09-14/CLAIMS.md#api-07) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [E01](WORK_ITEMS.md#e01) | Plan regression and repair |
| [API-08](../../../audits/2026-09-14/CLAIMS.md#api-08) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [E01](WORK_ITEMS.md#e01) | Plan regression and repair |
| [API-09](../../../audits/2026-09-14/CLAIMS.md#api-09) | Feature / result correctness | High when exposed | P2 / confirmed-reproduced | [R02](WORK_ITEMS.md#r02) | Plan regression and repair |
| [API-10](../../../audits/2026-09-14/CLAIMS.md#api-10) | Feature / result correctness | High when exposed | P2 / confirmed-reproduced | [R02](WORK_ITEMS.md#r02) | Plan regression and repair |
| [API-11](../../../audits/2026-09-14/CLAIMS.md#api-11) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [R05](WORK_ITEMS.md#r05) | Plan regression and repair |
| [API-12](../../../audits/2026-09-14/CLAIMS.md#api-12) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [D02](WORK_ITEMS.md#d02) | Design/profile before optimization |
| [API-13](../../../audits/2026-09-14/CLAIMS.md#api-13) | Correctness / memory safety | High when exposed | P1 / confirmed-source | [S02](WORK_ITEMS.md#s02) | Plan regression and repair |
| [API-14](../../../audits/2026-09-14/CLAIMS.md#api-14) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [R03](WORK_ITEMS.md#r03) | Plan regression and repair |
| [API-15](../../../audits/2026-09-14/CLAIMS.md#api-15) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [R03](WORK_ITEMS.md#r03) | Plan regression and repair |
| [API-16](../../../audits/2026-09-14/CLAIMS.md#api-16) | Feature / result correctness | High when exposed | P2 / confirmed-reproduced | [R03](WORK_ITEMS.md#r03) | Plan regression and repair |
| [API-17](../../../audits/2026-09-14/CLAIMS.md#api-17) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [R04](WORK_ITEMS.md#r04) | Plan regression and repair |
| [API-18](../../../audits/2026-09-14/CLAIMS.md#api-18) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [S03](WORK_ITEMS.md#s03) | Plan regression and repair |
| [API-19](../../../audits/2026-09-14/CLAIMS.md#api-19) | Feature / result correctness | Low / limited exposure | P3 / confirmed-reproduced | [S02](WORK_ITEMS.md#s02) | Plan regression and repair |
| [API-20](../../../audits/2026-09-14/CLAIMS.md#api-20) | Feature / result correctness | Low / limited exposure | P3 / confirmed-reproduced | [L02](WORK_ITEMS.md#l02) | Plan regression and repair |
| [API-21](../../../audits/2026-09-14/CLAIMS.md#api-21) | Correctness / memory safety | Medium | P2 / confirmed-source | [Q03](WORK_ITEMS.md#q03) | Plan regression and repair |
| [API-22](../../../audits/2026-09-14/CLAIMS.md#api-22) | Accuracy / scale limit | High when exposed | P2 / qualified | [P02](WORK_ITEMS.md#p02) | Resolve contract/scope before change |
| [API-23](../../../audits/2026-09-14/CLAIMS.md#api-23) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [P01](WORK_ITEMS.md#p01) | Design/profile before optimization |
| [API-24](../../../audits/2026-09-14/CLAIMS.md#api-24) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [P01](WORK_ITEMS.md#p01) | Design/profile before optimization |
| [API-25](../../../audits/2026-09-14/CLAIMS.md#api-25) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [P02](WORK_ITEMS.md#p02) | Design/profile before optimization |
| [API-26](../../../audits/2026-09-14/CLAIMS.md#api-26) | Feature / result correctness | Low / limited exposure | P3 / qualified | [R04](WORK_ITEMS.md#r04) | Resolve contract/scope before change |
| [API-27](../../../audits/2026-09-14/CLAIMS.md#api-27) | Resource / availability | Low / limited exposure | P3 / qualified | [E02](WORK_ITEMS.md#e02) | Resolve contract/scope before change |
| [API-28](../../../audits/2026-09-14/CLAIMS.md#api-28) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [E02](WORK_ITEMS.md#e02) | Design/profile before optimization |
| [API-29](../../../audits/2026-09-14/CLAIMS.md#api-29) | Resource / availability | High when exposed | P2 / confirmed-source | [E05](WORK_ITEMS.md#e05) | Plan regression and repair |
| [API-30](../../../audits/2026-09-14/CLAIMS.md#api-30) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [D02](WORK_ITEMS.md#d02) | Plan regression and repair |
| [API-31](../../../audits/2026-09-14/CLAIMS.md#api-31) | Feature / result correctness | Low / limited exposure | P3 / confirmed-reproduced | [D03](WORK_ITEMS.md#d03) | Plan regression and repair |
| [API-32](../../../audits/2026-09-14/CLAIMS.md#api-32) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [E04](WORK_ITEMS.md#e04) | Design/profile before optimization |
| [API-33](../../../audits/2026-09-14/CLAIMS.md#api-33) | Feature / result correctness | Medium | P1 / confirmed-source | [I01](WORK_ITEMS.md#i01) | Plan regression and repair |
| [API-34](../../../audits/2026-09-14/CLAIMS.md#api-34) | Feature / result correctness | Medium | P2 / confirmed-source | [D01](WORK_ITEMS.md#d01) | Plan regression and repair |
| [API-35](../../../audits/2026-09-14/CLAIMS.md#api-35) | Contract / integration | Not a separate defect | P3 / qualified | [D01](WORK_ITEMS.md#d01) | Preserve contract/control |
| [API-36](../../../audits/2026-09-14/CLAIMS.md#api-36) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [D01](WORK_ITEMS.md#d01) | Design/profile before optimization |
| [API-37](../../../audits/2026-09-14/CLAIMS.md#api-37) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [E04](WORK_ITEMS.md#e04) | Plan regression and repair |
| [API-38](../../../audits/2026-09-14/CLAIMS.md#api-38) | Contract / integration | Not a separate defect | P3 / qualified | [D02](WORK_ITEMS.md#d02) | Preserve contract/control |
| [API-39](../../../audits/2026-09-14/CLAIMS.md#api-39) | Contract / integration | Medium | P2 / qualified | [N01](WORK_ITEMS.md#n01) | Resolve contract/scope before change |
| [API-40](../../../audits/2026-09-14/CLAIMS.md#api-40) | Contract / integration | Not a separate defect | P3 / confirmed-source | [D01](WORK_ITEMS.md#d01) | Preserve contract/control |
| [API-41](../../../audits/2026-09-14/CLAIMS.md#api-41) | Feature / result correctness | Low / limited exposure | P3 / confirmed-source | [P02](WORK_ITEMS.md#p02) | Plan regression and repair |
| [API-42](../../../audits/2026-09-14/CLAIMS.md#api-42) | Contract / integration | Low / limited exposure | P3 / qualified | [E02](WORK_ITEMS.md#e02) | Resolve contract/scope before change |
| [API-43](../../../audits/2026-09-14/CLAIMS.md#api-43) | Investigation | Undetermined | P3 / unverified | [V04](WORK_ITEMS.md#v04) | Investigate before deciding |
| [API-44](../../../audits/2026-09-14/CLAIMS.md#api-44) | Feature / result correctness | Medium | P2 / confirmed-source | [E01](WORK_ITEMS.md#e01) | Plan regression and repair |
| [API-45](../../../audits/2026-09-14/CLAIMS.md#api-45) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [P01](WORK_ITEMS.md#p01) | Design/profile before optimization |
| [API-46](../../../audits/2026-09-14/CLAIMS.md#api-46) | Feature / result correctness | Low / limited exposure | P3 / confirmed-source | [R01](WORK_ITEMS.md#r01) | Plan regression and repair |
| [API-47](../../../audits/2026-09-14/CLAIMS.md#api-47) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [O01](WORK_ITEMS.md#o01) | Plan regression and repair |
| [API-48](../../../audits/2026-09-14/CLAIMS.md#api-48) | Contract / integration | Not a separate defect | P3 / qualified | [E02](WORK_ITEMS.md#e02) | Preserve contract/control |
| [NUM-01](../../../audits/2026-09-14/CLAIMS.md#num-01) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [N01](WORK_ITEMS.md#n01) | Plan regression and repair |
| [NUM-02](../../../audits/2026-09-14/CLAIMS.md#num-02) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [N01](WORK_ITEMS.md#n01) | Plan regression and repair |
| [NUM-03](../../../audits/2026-09-14/CLAIMS.md#num-03) | Feature / result correctness | High when exposed | P2 / confirmed-reproduced | [Q01](WORK_ITEMS.md#q01) | Plan regression and repair |
| [NUM-04](../../../audits/2026-09-14/CLAIMS.md#num-04) | Feature / result correctness | Medium | P2 / confirmed-source | [Q01](WORK_ITEMS.md#q01) | Plan regression and repair |
| [NUM-05](../../../audits/2026-09-14/CLAIMS.md#num-05) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [Q01](WORK_ITEMS.md#q01) | Plan regression and repair |
| [NUM-06](../../../audits/2026-09-14/CLAIMS.md#num-06) | Feature / result correctness | High when exposed | P2 / confirmed-reproduced | [N03](WORK_ITEMS.md#n03) | Plan regression and repair |
| [NUM-07](../../../audits/2026-09-14/CLAIMS.md#num-07) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [N04](WORK_ITEMS.md#n04) | Plan regression and repair |
| [NUM-08](../../../audits/2026-09-14/CLAIMS.md#num-08) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [L01](WORK_ITEMS.md#l01) | Plan regression and repair |
| [NUM-09](../../../audits/2026-09-14/CLAIMS.md#num-09) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [L01](WORK_ITEMS.md#l01) | Plan regression and repair |
| [NUM-10](../../../audits/2026-09-14/CLAIMS.md#num-10) | Feature / result correctness | Low / limited exposure | P3 / qualified | [L02](WORK_ITEMS.md#l02) | Resolve contract/scope before change |
| [NUM-11](../../../audits/2026-09-14/CLAIMS.md#num-11) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [L03](WORK_ITEMS.md#l03) | Plan regression and repair |
| [NUM-12](../../../audits/2026-09-14/CLAIMS.md#num-12) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [L03](WORK_ITEMS.md#l03) | Plan regression and repair |
| [NUM-13](../../../audits/2026-09-14/CLAIMS.md#num-13) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [L03](WORK_ITEMS.md#l03) | Plan regression and repair |
| [NUM-14](../../../audits/2026-09-14/CLAIMS.md#num-14) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [N02](WORK_ITEMS.md#n02) | Plan regression and repair |
| [NUM-15](../../../audits/2026-09-14/CLAIMS.md#num-15) | Contract / integration | Medium | P2 / qualified | [N02](WORK_ITEMS.md#n02) | Resolve contract/scope before change |
| [NUM-16](../../../audits/2026-09-14/CLAIMS.md#num-16) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [Q02](WORK_ITEMS.md#q02) | Plan regression and repair |
| [NUM-17](../../../audits/2026-09-14/CLAIMS.md#num-17) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [Q02](WORK_ITEMS.md#q02) | Plan regression and repair |
| [NUM-18](../../../audits/2026-09-14/CLAIMS.md#num-18) | Feature / result correctness | Low / limited exposure | P3 / confirmed-reproduced | [Q03](WORK_ITEMS.md#q03) | Plan regression and repair |
| [NUM-19](../../../audits/2026-09-14/CLAIMS.md#num-19) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [Q02](WORK_ITEMS.md#q02) | Design/profile before optimization |
| [NUM-20](../../../audits/2026-09-14/CLAIMS.md#num-20) | Contract / integration | Low / limited exposure | P3 / qualified | [Q02](WORK_ITEMS.md#q02) | Resolve contract/scope before change |
| [NUM-21](../../../audits/2026-09-14/CLAIMS.md#num-21) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [Q02](WORK_ITEMS.md#q02) | Design/profile before optimization |
| [NUM-22](../../../audits/2026-09-14/CLAIMS.md#num-22) | Performance / memory | Medium | P2 / confirmed-source | [Q04](WORK_ITEMS.md#q04) | Plan regression and repair |
| [NUM-23](../../../audits/2026-09-14/CLAIMS.md#num-23) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [V04](WORK_ITEMS.md#v04) | Plan regression and repair |
| [NUM-24](../../../audits/2026-09-14/CLAIMS.md#num-24) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [Q02](WORK_ITEMS.md#q02) | Design/profile before optimization |
| [NUM-25](../../../audits/2026-09-14/CLAIMS.md#num-25) | Contract / integration | Low / limited exposure | P3 / qualified | [L03](WORK_ITEMS.md#l03) | Resolve contract/scope before change |
| [NUM-26](../../../audits/2026-09-14/CLAIMS.md#num-26) | Feature / result correctness | Low / limited exposure | P3 / qualified | [L03](WORK_ITEMS.md#l03) | Resolve contract/scope before change |
| [NUM-27](../../../audits/2026-09-14/CLAIMS.md#num-27) | Contract / integration | Not a separate defect | P3 / qualified | [L03](WORK_ITEMS.md#l03) | Preserve contract/control |
| [NUM-28](../../../audits/2026-09-14/CLAIMS.md#num-28) | Contract / integration | Low / limited exposure | P3 / qualified | [L03](WORK_ITEMS.md#l03) | Resolve contract/scope before change |
| [NUM-29](../../../audits/2026-09-14/CLAIMS.md#num-29) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [L04](WORK_ITEMS.md#l04) | Design/profile before optimization |
| [NUM-30](../../../audits/2026-09-14/CLAIMS.md#num-30) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [L04](WORK_ITEMS.md#l04) | Design/profile before optimization |
| [NUM-31](../../../audits/2026-09-14/CLAIMS.md#num-31) | Resource / availability | Low / limited exposure | P3 / qualified | [L04](WORK_ITEMS.md#l04) | Resolve contract/scope before change |
| [NUM-32](../../../audits/2026-09-14/CLAIMS.md#num-32) | Contract / integration | Not a separate defect | P3 / qualified | [D02](WORK_ITEMS.md#d02) | Preserve contract/control |
| [NUM-33](../../../audits/2026-09-14/CLAIMS.md#num-33) | Contract / integration | Not a separate defect | P3 / qualified | [D02](WORK_ITEMS.md#d02) | Preserve contract/control |
| [NUM-34](../../../audits/2026-09-14/CLAIMS.md#num-34) | Performance / memory | Low / limited exposure | P3 / qualified | [D02](WORK_ITEMS.md#d02) | Resolve contract/scope before change |
| [NUM-35](../../../audits/2026-09-14/CLAIMS.md#num-35) | Contract / integration | Low / limited exposure | P3 / qualified | [L01](WORK_ITEMS.md#l01) | Resolve contract/scope before change |
| [OWN-01](../../../audits/2026-09-14/CLAIMS.md#own-01) | Correctness / memory safety | High when exposed | P1 / confirmed-reproduced | [S01](WORK_ITEMS.md#s01) | Plan regression and repair |
| [OWN-02](../../../audits/2026-09-14/CLAIMS.md#own-02) | Correctness / memory safety | High when exposed | P1 / confirmed-reproduced | [S01](WORK_ITEMS.md#s01) | Plan regression and repair |
| [OWN-03](../../../audits/2026-09-14/CLAIMS.md#own-03) | Feature / result correctness | High when exposed | P1 / confirmed-reproduced | [I01](WORK_ITEMS.md#i01) | Plan regression and repair |
| [OWN-04](../../../audits/2026-09-14/CLAIMS.md#own-04) | Feature / result correctness | High when exposed | P1 / confirmed-reproduced | [I02](WORK_ITEMS.md#i02) | Plan regression and repair |
| [OWN-05](../../../audits/2026-09-14/CLAIMS.md#own-05) | Feature / result correctness | High when exposed | P1 / confirmed-reproduced | [O01](WORK_ITEMS.md#o01) | Plan regression and repair |
| [OWN-06](../../../audits/2026-09-14/CLAIMS.md#own-06) | Correctness / memory safety | Medium | P2 / confirmed-reproduced | [S02](WORK_ITEMS.md#s02) | Plan regression and repair |
| [OWN-07](../../../audits/2026-09-14/CLAIMS.md#own-07) | Performance / memory | Medium | P2 / confirmed-reproduced | [P01](WORK_ITEMS.md#p01) | Plan regression and repair |
| [OWN-08](../../../audits/2026-09-14/CLAIMS.md#own-08) | Correctness / memory safety | Medium | P2 / confirmed-reproduced | [O04](WORK_ITEMS.md#o04) | Plan regression and repair |
| [OWN-09](../../../audits/2026-09-14/CLAIMS.md#own-09) | Correctness / memory safety | Medium | P2 / qualified | [O04](WORK_ITEMS.md#o04) | Resolve contract/scope before change |
| [OWN-10](../../../audits/2026-09-14/CLAIMS.md#own-10) | Resource / availability | Medium | P2 / confirmed-reproduced | [O01](WORK_ITEMS.md#o01) | Plan regression and repair |
| [OWN-11](../../../audits/2026-09-14/CLAIMS.md#own-11) | Correctness / memory safety | Medium | P2 / confirmed-reproduced | [O04](WORK_ITEMS.md#o04) | Plan regression and repair |
| [OWN-12](../../../audits/2026-09-14/CLAIMS.md#own-12) | Performance / memory | High when exposed | P2 / confirmed-source | [O03](WORK_ITEMS.md#o03) | Plan regression and repair |
| [OWN-13](../../../audits/2026-09-14/CLAIMS.md#own-13) | Performance / memory | High when exposed | P2 / confirmed-reproduced | [O02](WORK_ITEMS.md#o02) | Plan regression and repair |
| [OWN-14](../../../audits/2026-09-14/CLAIMS.md#own-14) | Feature / result correctness | Low / limited exposure | P3 / confirmed-reproduced | [O02](WORK_ITEMS.md#o02) | Plan regression and repair |
| [OWN-15](../../../audits/2026-09-14/CLAIMS.md#own-15) | Correctness / memory safety | Low / limited exposure | P3 / confirmed-source | [O05](WORK_ITEMS.md#o05) | Plan regression and repair |
| [OWN-16](../../../audits/2026-09-14/CLAIMS.md#own-16) | Correctness / memory safety | Low / limited exposure | P3 / confirmed-source | [O05](WORK_ITEMS.md#o05) | Plan regression and repair |
| [OWN-17](../../../audits/2026-09-14/CLAIMS.md#own-17) | Correctness / memory safety | Medium | P2 / qualified | [O05](WORK_ITEMS.md#o05) | Resolve contract/scope before change |
| [OWN-18](../../../audits/2026-09-14/CLAIMS.md#own-18) | Feature / result correctness | Low / limited exposure | P3 / confirmed-source | [I02](WORK_ITEMS.md#i02) | Plan regression and repair |
| [OWN-19](../../../audits/2026-09-14/CLAIMS.md#own-19) | Performance / memory | Medium | P2 / confirmed-source | [I02](WORK_ITEMS.md#i02) | Design/profile before optimization |
| [OWN-20](../../../audits/2026-09-14/CLAIMS.md#own-20) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [D02](WORK_ITEMS.md#d02) | Design/profile before optimization |
| [OWN-21](../../../audits/2026-09-14/CLAIMS.md#own-21) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [D02](WORK_ITEMS.md#d02) | Design/profile before optimization |
| [OWN-22](../../../audits/2026-09-14/CLAIMS.md#own-22) | Performance / memory | Low / limited exposure | P3 / confirmed-source | [O03](WORK_ITEMS.md#o03) | Design/profile before optimization |
| [OWN-23](../../../audits/2026-09-14/CLAIMS.md#own-23) | Contract / integration | Low / limited exposure | P3 / qualified | [V04](WORK_ITEMS.md#v04) | Resolve contract/scope before change |
| [OWN-24](../../../audits/2026-09-14/CLAIMS.md#own-24) | Contract / integration | Not a separate defect | Context / qualified | [O05](WORK_ITEMS.md#o05) | Preserve contract/control |
| [OWN-25](../../../audits/2026-09-14/CLAIMS.md#own-25) | Investigation | Undetermined | P3 / unverified | [O04](WORK_ITEMS.md#o04) | Investigate before deciding |
| [OWN-26](../../../audits/2026-09-14/CLAIMS.md#own-26) | Rejected inference | Not a separate defect | Context / rejected | [O03](WORK_ITEMS.md#o03) | No repair; retain negative control |
| [OWN-27](../../../audits/2026-09-14/CLAIMS.md#own-27) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [D02](WORK_ITEMS.md#d02) | Design/profile before optimization |
| [SER-01](../../../audits/2026-09-14/CLAIMS.md#ser-01) | Feature / result correctness | Medium | P2 / confirmed-reproduced | [B01](WORK_ITEMS.md#b01) | Plan regression and repair |
| [SER-02](../../../audits/2026-09-14/CLAIMS.md#ser-02) | Validation / reporting | Medium | P2 / confirmed-reproduced | [B01](WORK_ITEMS.md#b01) | Plan regression and repair |
| [SER-03](../../../audits/2026-09-14/CLAIMS.md#ser-03) | Performance / memory | Low / limited exposure | P3 / qualified | [B01](WORK_ITEMS.md#b01) | Resolve contract/scope before change |
| [SER-04](../../../audits/2026-09-14/CLAIMS.md#ser-04) | Contract / integration | Low / limited exposure | P3 / qualified | [B01](WORK_ITEMS.md#b01) | Resolve contract/scope before change |
| [SER-05](../../../audits/2026-09-14/CLAIMS.md#ser-05) | Contract / integration | Low / limited exposure | P3 / confirmed-source | [B01](WORK_ITEMS.md#b01) | Design/profile before optimization |
| [SER-06](../../../audits/2026-09-14/CLAIMS.md#ser-06) | Contract / integration | Low / limited exposure | P3 / confirmed-reproduced | [B01](WORK_ITEMS.md#b01) | Plan regression and repair |
| [SER-07](../../../audits/2026-09-14/CLAIMS.md#ser-07) | Context / guardrail | Not a separate defect | P3 / confirmed-reproduced | [B01](WORK_ITEMS.md#b01) | Preserve contract/control |
| [SER-08](../../../audits/2026-09-14/CLAIMS.md#ser-08) | Context / guardrail | Not a separate defect | P2 / confirmed-source | [S03](WORK_ITEMS.md#s03) | Preserve contract/control |
| [TOOL-01](../../../audits/2026-09-14/CLAIMS.md#tool-01) | Validation / reporting | High when exposed | P1 / confirmed-reproduced | [V01](WORK_ITEMS.md#v01) | Plan regression and repair |
| [TOOL-02](../../../audits/2026-09-14/CLAIMS.md#tool-02) | Validation / reporting | Medium | P2 / confirmed-reproduced | [V01](WORK_ITEMS.md#v01) | Plan regression and repair |
| [TOOL-03](../../../audits/2026-09-14/CLAIMS.md#tool-03) | Validation / reporting | Medium | P2 / confirmed-reproduced | [V02](WORK_ITEMS.md#v02) | Plan regression and repair |
| [TOOL-04](../../../audits/2026-09-14/CLAIMS.md#tool-04) | Validation / reporting | Medium | P2 / confirmed-source | [V03](WORK_ITEMS.md#v03) | Plan regression and repair |
| [TOOL-05](../../../audits/2026-09-14/CLAIMS.md#tool-05) | Validation / reporting | Medium | P2 / confirmed-reproduced | [V02](WORK_ITEMS.md#v02) | Plan regression and repair |
| [TOOL-06](../../../audits/2026-09-14/CLAIMS.md#tool-06) | Validation / reporting | Medium | P2 / confirmed-reproduced | [V02](WORK_ITEMS.md#v02) | Plan regression and repair |
| [TOOL-07](../../../audits/2026-09-14/CLAIMS.md#tool-07) | Validation / reporting | Medium | P2 / qualified | [V04](WORK_ITEMS.md#v04) | Resolve contract/scope before change |
| [TOOL-08](../../../audits/2026-09-14/CLAIMS.md#tool-08) | Validation / reporting | Low / limited exposure | P3 / confirmed-reproduced | [V04](WORK_ITEMS.md#v04) | Plan regression and repair |
| [TOOL-09](../../../audits/2026-09-14/CLAIMS.md#tool-09) | Validation / reporting | Medium | P2 / confirmed-source | [V02](WORK_ITEMS.md#v02) | Design/profile before optimization |
| [TOOL-10](../../../audits/2026-09-14/CLAIMS.md#tool-10) | Validation / reporting | Medium | P2 / qualified | [P02](WORK_ITEMS.md#p02) | Resolve contract/scope before change |
| [TOOL-11](../../../audits/2026-09-14/CLAIMS.md#tool-11) | Validation / reporting | Medium | P2 / confirmed-source | [V03](WORK_ITEMS.md#v03) | Plan regression and repair |
| [TOOL-12](../../../audits/2026-09-14/CLAIMS.md#tool-12) | Validation / reporting | Medium | P2 / confirmed-source | [V02](WORK_ITEMS.md#v02) | Design/profile before optimization |
| [TOOL-13](../../../audits/2026-09-14/CLAIMS.md#tool-13) | Investigation | Undetermined | P3 / unverified | [D02](WORK_ITEMS.md#d02) | Investigate before deciding |

## Claim and exposure notes

These short notes retain the limits that matter when assigning severity. They do not replace the reproduction record.

### API-01

**Two search facades replace unsupported metric witnesses with Euclidean**

- Exposure: Public TopKSelection.nearest/batchNearest and default collection search; Operations respects requested metric
- Qualification: Collection probe itself shows wrong score, not rank reversal; source proves reversal for search_probe candidate set. Not a downstream VectorIndex implementation bug.

### API-02

**Batch configuration storage is disconnected from operation settings**

- Exposure: Public BatchOperations; matrix flags alone use live settings
- Qualification: findNearest uses ParallelHeuristic, not configurable general parallelThreshold. Do not imply every knob governs every API.

### API-03

**Configuration read-modify-write loses concurrent updates**

- Exposure: Public updateConfiguration
- Qualification: Actor individual access remains isolated; failure is transaction lost update, not unsynchronized stored memory.

### API-04

**Several schedulers eagerly create all child tasks**

- Exposure: ComputeProvider defaults; public BatchOperations.process/map/filter; public Dim768 cosineSimilarityMatrix and TriangularMatrix.buildParallel; downstream VA inherited batch defaults
- Qualification: Task count/resource exposure, not 64 simultaneous CPU threads or measured slowdown. CPUComputeProvider own bounded chunks/nesting fix is intact. Default reduction is coarsely chunked, not per-item.

### API-05

**Cached raw actor pool allocations are not freed on owner destruction**

- Exposure: Public scoped SwiftBufferPool and TypedBufferPool; internal BufferPool same pattern
- Qualification: Proof covers returned cached allocation without active handles. It does not establish active-handle teardown policy, generic element ownership, or leak in separately repaired MemoryPool.

### API-06

**SwiftBufferPool age cleanup does not enforce maxPoolSize**

- Exposure: Public SwiftBufferPool opt-in; Operations buffer binding currently unused internally
- Qualification: No universal RSS/time claim; indefinite idle retention is source-derived. maxBufferSize remains enforced on return.

### API-07

**Nonpositive process batch sizes trap or discard work**

- Exposure: Public throwing BatchOperations.process explicit parameter
- Qualification: Int positivity is a reasonable semantic expectation, but throwing API currently has neither clear contract nor validation; rank below ordinary positive-input bugs.

### API-08

**CPU chunk endpoints overflow before clamping valid short ranges**

- Exposure: Public explicitly parallel CPU helpers; default automatic only when gate reached; huge explicit minChunk option
- Qualification: No enormous allocation is required for CPU witnesses. Do not claim tiny near-max failure for default reduction without a case that actually overflows its smaller chunk.

### API-09

**Failed collection update deletes existing value**

- Exposure: Public MutableVectorCollection default; SimpleVectorCollection reference implementation
- Qualification: SimpleVectorCollection is a small/reference utility, not actual VectorIndex storage. Prechecking dimension alone does not guarantee rollback for arbitrary add failures.

### API-10

**Explicit/generated IDs overwrite records**

- Exposure: SimpleVectorCollection public insertion APIs
- Qualification: Reference collection scope; integer exhaustion is design edge, not replayed production failure.

### API-11

**computedScore pole breaks negative-dot relevance**

- Exposure: Public SearchResult fallback; actual collection searchDotProduct producer
- Qualification: No reviewed downstream computedScore callers; do not claim observed ANN/GPU regression. There is no universal metric-free transform for all Float scores.

### API-12

**Advertised buffer/global configuration hooks have no active internal consumers**

- Exposure: Public reserved-looking bindings and immutable facade
- Qualification: Provider coexistence is not itself a defect. ArraySIMDProvider and pointer SIMDProvider have different legitimate scopes; API map already discloses static-kernel limitations. Extension-only vector dispatch is owned by vector reviewer.

### API-13

**Collection aggregation trusts first row before unsafe reads/forced factories**

- Exposure: Public Operations.centroid/statistics; generic factory boundary
- Qualification: No equal-dimension precondition appears at these aggregation entries. Do not transfer MatrixDistance unsafe contracts to them. Pointer overread is source proof; garbage returned by an unsanitized probe is not proof of exact read extent.

### API-14

**Zero-query SoA cosine fills oversized output suffix**

- Exposure: Public Safe512 and dimension wrappers using shared internal routes
- Qualification: Writes remain inside provided capacity; this is logical-output overwrite, not out-of-allocation memory corruption. Nonzero branches write N only; no documented requirement to overwrite scratch suffix.

### API-15

**Pairwise hard-coded zero diagonal fails built-in negative dot**

- Exposure: Public BatchOperations.pairwiseDistances direct routes
- Qualification: Directed metric N2/N100 discrepancy reproduced but pairwise docs promise symmetric output: treat arbitrary asymmetric support as contract clarification, not required fix. VectorMath pairwise explicitly documents zero diagonal.

### API-16

**cosineSimilarityMatrix returns negative off-diagonal similarity and zero diagonal**

- Exposure: Public Array<Vector<Dim768>> convenience
- Qualification: Separate from directed-metric contract. Exact rounded -1.0000001 is numeric noise; sign/diagonal error is order-one.

### API-17

**Bulk bounded heap membership differs from streaming; negative capacity loops**

- Exposure: Public MinHeap opt-in; no evidence high-level Operations uses bulk constructor
- Qualification: Implicit nonnegative capacity is reasonable but missing validation; separate P3 invalid-parameter subcase from valid bounded-selection P2. Parent/root ordering does not make heap prefix correct top-k.

### API-18

**TopKResult decoder violates parallel-array invariant**

- Exposure: Public Codable from malformed external data
- Qualification: Malformed serialized input, not valid round-trip bug. Silent zip truncation is consequence of same invariant violation.

### API-19

**Factory batch dimension zero reaches remainder by zero**

- Exposure: Public throwing VectorTypeFactory.batch invalid dimension
- Qualification: A vector dimension normally implies positivity; classify invalid-parameter validation gap rather than ordinary valid-input failure. Not all factory methods trap at zero.

### API-20

**PCA oversampling Int.max overflows before bounded sketch clamp**

- Exposure: Public optional extreme PCAConfig; two valid one-dimensional samples
- Qualification: Extreme optional knob, not default PCA numeric failure or enormous input allocation.

### API-21

**FP16 diagnostics enabled flag is read without lock**

- Exposure: Concurrent diagnostics enable/disable with public mixed-precision conversions; opt-in recording
- Qualification: Source-proven conflicting access, no TSAN report. Recorders normally unreachable while always disabled; default entire library not racy from this flag.

### API-22

**Automatic batch GEMM requires full Q*N scores and can change near-neighbor ranking**

- Exposure: Operations optimized Euclidean/cosine Q>=8,N>=256 unless BatchKernelProvider installed
- Qualification: Already explicitly documented full allocation and cancellation/ranking limitation. Not new hidden bug or universally observed across hardware. Sequential provider and BatchOperations matrix setting do not bypass separate gate.

### API-23

**Small-k array/generic selection retains O(N) temporary distances/pairs**

- Exposure: Public TopKSelection arrays/nearest, generic Operations and BatchOperations; pointer small-k already streams
- Qualification: No measured speedup/RSS; tuple metadata not full copied vector payload in every path. Avoid Int32 count restriction when adopting pointer implementation.

### API-24

**Redundant matrix reshape, validation and oversized partial heaps increase peak work**

- Exposure: GEMM pairwise, fallback batch search, optimized near-full-k search
- Qualification: Source logical live allocations/complexity, not measured peak RSS after optimizer. C*k is upper-model for C concurrent chunks, not all possible schedulers.

### API-25

**MatrixDistance cannot consume downstream contiguous cache without packing**

- Exposure: Public MatrixDistance and VectorIndex centroid batch scorer
- Qualification: Existing unchecked shape/preparation/BLAS contracts are documented; do not recast malformed MatrixDistance inputs as new memory-safety findings. No copy latency measured.

### API-26

**KNearestHeap retains NaN ahead of numeric candidate**

- Exposure: Public legacy helper; only internal ExecutionOperations production caller
- Qualification: No documented NaN-last/index tie contract here; not regression of fixed Operations/TopKSelection or their tests.

### API-27

**Schedulers do not promise or enforce prompt cancellation inside CPU chunks**

- Exposure: Public schedulers with noncooperative synchronous body
- Qualification: Contract gap/opportunity, not proof task cancellation is universally ignored; throwing/suspending user closures can cooperate.

### API-28

**Public executeBatch autotuning claim contradicts single full-range invocation**

- Exposure: Public CPUComputeProvider.executeBatch direct caller
- Qualification: No measured runtime regression. Existing inline comment explicitly admits sequential implementation, contradicting public doc.

### API-29

**AutoTuning lock covers arbitrary calibrator and serializes unrelated keys**

- Exposure: Public calibrator; live Operations generic/custom metric calibration unless disabled/cache already populated/cap exhausted
- Qualification: Source proof, no deadlock subprocess run. Not all Operations routes calibrate: fused top-k and installed providers may bypass; cached keys bypass callback. No duration benchmark.

### API-30

**ComputeDevice runtime metadata includes placeholders**

- Exposure: Public read-only device queries, not actual automatic GPU discovery
- Qualification: CPU-only Core boundary is documented; placeholders cannot be used as actual device validation. No unsupported hardware runtime probe.

### API-31

**Disabled logger eagerly constructs messages; timer uses wall clock**

- Exposure: Public Logger and PerformanceTimer
- Qualification: No timing overhead or clock-jump failure measured; private minimumLevel field locking is sound.

### API-32

**MemoryPool recomputes pooled bytes using flatMap arrays under lock**

- Exposure: Repaired public MemoryPool on buffer return
- Qualification: Performance hypothesis only; not regression of prior retention fix or evidence lock contention dominates.

### API-33

**Normalization hints reach real downstream query and cached inverse norm consumers**

- Exposure: VectorIndex typed APIs; VA cosine kernels; Core wrapper defect owned separately
- Qualification: No downstream build/GPU replay. VA AcceleratedVectorIndex captures hints but its doc explicitly says L2 kernel does not yet consume them, so captured metadata alone is not evidence of active L2 corruption. Cached Index damage is conditional on live cache/route.

### API-34

**VectorIndex batch query flag comes only from first query**

- Exposure: VectorIndex Flat/IVF/optimized flat typed batchSearch; HNSW handles each query
- Qualification: Separate downstream defect even after Core A2 fix; no executed index integration test.

### API-35

**Provider seam erases Float scalar/score conventions**

- Exposure: Public BatchKernelProvider/VectorCollection; VA direct enum distance versus Core distance
- Qualification: Documented deliberate direct-API semantics and provider ordering variation, not proven transparent route bug. Float-constraining seam is design suggestion with compatibility cost.

### API-36

**VectorCollection is not actual VectorIndex conformance seam**

- Exposure: Core small/reference collection API and local downstream snapshots
- Qualification: Future-tense will implement is aspiration, but example presenting HNSW conformance does not compile against this snapshot. No claim production ANN storage belongs in Core.

### API-37

**BufferProvider example puts await in defer and handles require explicit lifetime policy**

- Exposure: Public raw BufferHandle and BufferProvider docs
- Qualification: No demonstrated valid-use UAF; VA requires keeping provider alive while handle active. Ownership/borrow docs elsewhere already state unsafe lifetime duties.

### API-38

**Public low-level implementation types create support obligations**

- Exposure: Exported caches/kernels/providers/platform helpers
- Qualification: Not correctness bug by visibility alone and no binary ABI stability inferred. Complete symbol graph inventory not performed here.

### API-39

**VA CPU fallbacks and Core unchecked normalization use different finite-range algorithms**

- Exposure: Active VA fallback normalization and acceleratedNormalizeUnchecked fallback to Core
- Qualification: Unchecked zero input violates explicit nonzero assertion and is not an additional valid-input defect. GPU parity not tested; avoid universal provider parity claim.

### API-40

**Local sibling dependency pins differ from reviewed Core main**

- Exposure: Local source snapshots; no compatibility build
- Qualification: Normal unreleased development, not version compatibility defect. VA stale comments about unavailable v26 enums contradict manifest, downstream docs only.

### API-41

**Existing provider tests do not cross GEMM query threshold**

- Exposure: BatchKernelProvider precedence tests
- Qualification: Test gap, not provider dispatch bug.

### API-42

**Dormant execution engines duplicate weak validation/reduction semantics**

- Exposure: Internal ExecutionOperations/SyncBatchOperations; live transform/matrix conveniences separately identified
- Qualification: Do not label all ExecutionOperations dormant: public Array transforms/cosine matrix are live. Nonidentity initial conflicts with identity reduction law in existing ComputeProvider tests, so law/documentation first.

### API-43

**SoA/unrolled/prefetch throughput claims lack supplied benchmark/compiler evidence**

- Exposure: Public comments/helpers and batch kernel docs
- Qualification: Unsupported speedup claims, not demonstrated slowdown; optimizer may remove nominal prefetch, assembly not inspected.

### API-44

**unrolledLoop arbitrary stride emits wrong indices; zero widths fail to progress**

- Exposure: Public helper opt-in; no internal/downstream production callers established
- Qualification: Original exact-source execution exists but was not independently replayed here; source one-step proof is sufficient. Zero/negative width subcase is lower-priority invalid-input validation; positive1/5/8 violates ordinary stride expectation.

### API-45

**VectorMath.nearestNeighbors fully sorts while claiming heap complexity**

- Exposure: Public fixed Vector convenience
- Qualification: Negative neighbor count has natural nonnegative precondition but missing explicit API documentation; not a new finite k wrong-ranking proof. Bare Float comparator has unspecified NaN/tie policy; do not transplant modern selector law without decision.

### API-46

**Named optimized metric factory cannot create dotProduct**

- Exposure: Public OptimizedDistanceMetrics.create(named:)
- Qualification: One-step mapping proof, not runtime replay. Separate from search metric witness replacement.

### API-47

**Singleton full-matrix triangular import reads nonexistent element**

- Exposure: Public TriangularMatrix.fromFullMatrix
- Qualification: Singleton input satisfies stated square-matrix requirement. Diagonal/lower-triangle omission is deliberate and is not the defect.

### API-48

**CPU sequential reduce ignores initial but identity is the existing provider law**

- Exposure: Public CPUComputeProvider.parallelReduce
- Qualification: Not a new numeric defect from nonidentity input. Current protocol comments should state identity and associativity requirements; floating-point nonassociativity means bitwise mode equivalence is not promised.

### NUM-01

**Scalar division computes an overflowing reciprocal even when each quotient is finite.**

- Exposure: Public vector operators, public Swift array/static providers; Accelerate provider also fails on this host.
- Qualification: Do not assume vDSP_vsdiv repairs the problem: it fails on this Apple host. Direct inPlaceDivide and /= are distinct observed routes.

### NUM-02

**Normalization reconstructs a mismatched scale or relies on an unrepresentable norm/reciprocal.**

- Exposure: Public generic checked normalization and public pointer path; concrete optimized helpers have additional scale limits.
- Qualification: Representability of the full norm or its reciprocal is unnecessary for a finite unit direction. Deep-subnormal unchanged behavior is currently an explicit implementation policy, not proof that normalization is mathematically impossible.

### NUM-03

**QuantizationSchemes clamps its zero point so positive/negative-only and constant intervals cannot be represented.**

- Exposure: Public opt-in QuantizationSchemes calibration.
- Qualification: Independent decode is legitimate: same formula appears in analyzeQuantizationError. LinearQuantizationParams is a separate public surface and its existing tests do not establish this one.

### NUM-04

**Clamping integer conversions trap before the clamp for finite large intermediates.**

- Exposure: Public scalar quantizer and asymmetric parameter construction.
- Qualification: Keep finite outliers distinct from NaN/Inf policy and from intentionally unsafe raw buffers. An Int32 clamp avoids traps but can still destroy representation for high offsets.

### NUM-05

**Quantized cosine overflows the product of finite squared norms.**

- Exposure: Public INT8 methods at widths512/768/1536, opt-in.
- Qualification: Not quantization noise. sqrt(qMagSq*cMagSq) overflows even with finite individual squared norms. Fixing only square-root order still leaves scale-squared/product underflow/overflow elsewhere.

### NUM-06

**Batch magnitude variance uses cancellation-prone raw moments above N=1000.**

- Exposure: Public BatchOperations.statistics; serial route is also inaccurate for large repeated values.
- Qualification: Statistics concerns magnitudes, not component covariance. At1e18 adding1 rounds away, so those inputs really are identical. Merely replacing negative variance clamp does not repair cancellation or squared-sum overflow.

### NUM-07

**Entropy cutoff and Float sum violate finite scale invariance.**

- Exposure: Public Vector entropy facade, small and large routes.
- Qualification: Existing tiny-probability cutoff beside a dominant1 does not justify the absolute absSum cutoff on uniform vectors.

### NUM-08

**Swift SVD accepts nonorthogonal columns as converged at harmless input scales.**

- Exposure: Public SwiftLinearAlgebraProvider; PCA when override selects it, default on unsupported LAPACK platforms.
- Qualification: Full rank and condition number2.618: documented null-space-basis limitation is irrelevant. Reconstruction alone can pass with invalid SVD factors.

### NUM-09

**Swift QR and SVD final normalization lose representable finite vectors through Float squared norms.**

- Exposure: Public Swift fallback and PCA using its QR.
- Qualification: Widened norm accumulation immediately narrowed into unsafe reciprocal/alpha-beta arithmetic is insufficient for full exponent range.

### NUM-10

**Rounding PCA mean before centering doubles variance in a valid offset fixture.**

- Exposure: Default LAPACK-backed centered PCA, large-offset near-constant data.
- Qualification: Numerical-quality improvement under broad documented conditioning limitations, not universal relative-accuracy violation. Error term is n/(n-1)*deltaMean*deltaMean^T before further rounding.

### NUM-11

**UMAP output curve fit depends on units and silently returns its initial guess.**

- Exposure: Public valid minDist/spread configurations through internal fitter.
- Qualification: Claim concerns kernel calibration, not stochastic layout-scale invariance: learning rate, gradient clipping and init need not rescale the whole layout.

### NUM-12

**Smooth-kNN bandwidth search starts at absolute1 and fails outside its fixed64-step range.**

- Exposure: Public graph-only UMAP input with finite nonnegative distances.
- Qualification: Use solvable rows for equation assertions; duplicates/tied minimum can make target unattainable and are already handled by floor. Float sigma can itself become unrepresentable at extreme inputs even after bracket fix.

### NUM-13

**UMAP accepts infinite learning rate/spread at an existing throwing validation boundary.**

- Exposure: Public UMAPConfig used by both overloads.
- Qualification: NaN generally fails current comparisons. Finiteness checks are necessary but cannot promise every huge finite learning rate yields a finite converged layout.

### NUM-14

**Approximate equality accepts NaN differences as equal.**

- Exposure: Public VectorProtocol approximate equality; PrecisionValidator is internal dormant helper.
- Qualification: Matching infinities could be an intentional equality policy; NaN-versus-finite acceptance cannot be justified by tolerance. Negative/NaN/infinite tolerances also need explicit policy.

### NUM-15

**safeLog bypasses options for NaN and positive infinity.**

- Exposure: Public Vector safeLog; narrower method docs conflict with general options.
- Qualification: Behavior reproduced, but method-specific options documentation says non-positive values. Classify as inconsistent option semantics requiring resolution, rather than an unqualified promise of finite logarithm for every call.

### NUM-16

**Adaptive distance ignores threshold and candidate range.**

- Exposure: Explicit public adaptive API; not default Operations routing.
- Qualification: Default docs disclaim universal FP16 score bounds, but ignored threshold is still ineffective API control. Representability checks on both operands alone do not enforce an arbitrary relative error bound.

### NUM-17

**Quantized tuner cache excludes accuracyRequirement, making selection call-history dependent.**

- Exposure: Public opt-in singleton tuner; no automatic production consumer established.
- Qualification: Counterintuitive heuristic direction is not independently a bug until accuracyRequirement units/meaning are specified. Quantized tuner lock protects dictionary access; no race alleged here.

### NUM-18

**Diagnostics misclassifies underflows, rounded boundary values, and observed extrema.**

- Exposure: Opt-in diagnostics reached by public FP16 construction.
- Qualification: Swapping branches or using65504 as an overflow cutoff is insufficient; classification must follow actual rounded conversion.

### NUM-19

**KernelAutoTuner accepts memory/latency constraints it never uses and persists profiles without provenance.**

- Exposure: Public opt-in tuner, not automatic Operations dispatch.
- Qualification: Missing hardware identity is stale-profile risk, not reproduced bad dispatch; memoryFootprint is an estimate, not allocation measurement. Tests explicitly bypass profile selection under detected CI/XCTest, so existing passes do not cover ordinary persistence path.

### NUM-20

**Explicit selectMixedPrecisionStrategy maps full precision mode to nonzero tolerance.**

- Exposure: Public opt-in low-level tuner method.
- Qualification: Do not claim normal fp32Full dispatch automatically quantizes; contradictory behavior is limited to explicit mixed-selection entry point, whose docs expect cpuMixedPrecision workload.

### NUM-21

**Mixed tuner calibrates all5 strategies synchronously on exact workload key and converts candidates inside each reduced execution.**

- Exposure: First opt-in MixedPrecisionAutoTuner selection for a fresh dimension/count/accuracy key.
- Qualification: 125 is normal-path count, not unconditional exact total. Profiles use synthetic data and end-to-end conversion cost; they cannot establish arbitrary input accuracy or preconverted serving latency. No wall-clock regression measured.

### NUM-22

**FP16 range wrappers map all candidates even for small or empty ranges.**

- Exposure: Nine public range FP16/mixed wrappers.
- Qualification: No measured latency/allocation count; optimizer may influence realized overhead. Merely slicing without adjusting range/global output conventions can break indexing.

### NUM-23

**Source numerical/performance assurances exceed tests or contradict implementation.**

- Exposure: Public API comments and internal optimization documentation.
- Qualification: Do not convert unsupported speed claims into claims of measured slowdown; claims of external GPU limitations remain unverified without GPU execution.

### NUM-24

**Immutable FP16 vector pool cannot refill storage for conversion of new values via a supported API.**

- Exposure: Public opt-in pool.
- Qualification: It can reuse immutable values for repeated computation; claim is specifically unsupported reuse as mutable conversion destination. No allocation speed benefit measured.

### NUM-25

**CSR initializer may dereference malformed offsets before throwing validation catches them.**

- Exposure: Malformed public KNNGraph input outside documented structural caller obligation.
- Qualification: Not supported valid graph input; docs explicitly prohibit treating constructor as arbitrary parser, while source promises malformed structure throws. Frame as boundary/documentation inconsistency, not ordinary-use corruption.

### NUM-26

**Accepted extreme integer configuration can trap before throwing or clamping.**

- Exposure: Explicit oversized positive PCA/UMAP options on tiny ordinary data.
- Qualification: Operationally impractical settings; not ordinary-sized failure or giant allocation test. General allocation-safety disclaimer does not explain away the easily checked preallocation addition, but priority is below valid ordinary-input defects. Original UMAP report line229/241 corrected to215/225 for n*dimensions.

### NUM-27

**Exact brute-force kNN builder is exhaustive over computed Gram scores, not a reliable numerical distance oracle.**

- Exposure: Public reference KNNGraph.bruteForce.
- Qualification: Documented Gram cancellation elsewhere; do not present as novel UMAP algorithm defect. Broad numerical docs apply; exact label should not imply stable reference scores.

### NUM-28

**UMAP initial coordinates are shape checked, not validated for finiteness or safe scale.**

- Exposure: Explicit caller-supplied initialization.
- Qualification: No all-finite or conditioning guarantee; do not generalize to ordinary random/PCA initialization.

### NUM-29

**UMAP and PCA expose material in-memory storage costs at large corpus/sample size.**

- Exposure: Public in-memory UMAP/PCA; default sizes/workloads determine exposure.
- Qualification: F/R depend on reciprocity/pruning; staging/sort slack and ARC lifetimes vary. This is a byte model, not measured RSS. In-memory design explicitly documented.

### NUM-30

**Graph sorts/epoch scans and PCA per-point transforms offer concrete allocation/work opportunities.**

- Exposure: Public synchronous algorithms; no throughput regression measured.
- Qualification: Not missing ANN/GPU functionality; serial deterministic and in-memory contract is explicit.

### NUM-31

**Long synchronous manifold/factorization work lacks cancellation checkpoints.**

- Exposure: Callers using synchronous APIs, including inside async tasks.
- Qualification: No broken cancellation guarantee is documented. Large positive powerIterations requests more work intentionally.

### NUM-32

**C INT8 accumulation can exceed signed32 range only beyond supported fixed-vector dimensions.**

- Exposure: Generic exported C lanes API; optional Swift C INT8 call sites use fixed512/768/1536.
- Qualification: Do not allege active fixed-width overflow. C intrinsic lane behavior differs from scalar C signed-overflow language semantics; no Intel/NEON overflow execution or UBSan run.

### NUM-33

**C header range indexing wording and proposed unimplemented declarations need precise public status.**

- Exposure: Public C/header and Swift wrapper interface; no current Swift link failure established.
- Qualification: Header declaration of proposed symbol is API-hygiene risk, not proof a normal build fails.

### NUM-34

**Allocation-heavy vector helpers, unequal SIMD reduction structure and platform heuristics warrant measurement, not speed claims.**

- Exposure: Public VectorMath helpers; internal PlatformConfiguration and mostly unused optimization helpers.
- Qualification: No universal measured slowdown; no illegal instruction dispatch is proven from these width/cache hints.

### NUM-35

**LAPACK dimension ceiling does not by itself establish extreme workspace arithmetic safety.**

- Exposure: Shapes far beyond documented small-panel operating envelope.
- Qualification: Do not elevate to verified practical memory-corruption defect. Two Int32 dimensions multiply safely in64-bit Swift Int; workspace arithmetic is distinct.

### OWN-01

**Public DynamicVector and optimized FP32 storage mutations break shape invariants.**

- Exposure: public default representations
- Qualification: Do not say every vector or every check fails: generic Vector<D> uses a count-constrained DimensionStorage; FP16 conversions explicitly check malformed FP32 storage. No out-of-bounds write was executed. Different hash values are an observed consequence, not a mathematical guarantee against incidental hash collisions.

### OWN-02

**INT8 raw constructors, mutable storage, and synthesized Codable permit malformed fixed storage.**

- Exposure: public opt-in INT8 APIs
- Qualification: FP16 vectors are immutable and exact-length validated; they are not part of this shape defect. Invalid INT8 storage is observed; resulting unsafe reads are source-derived, not an executed memory fault. Synthesized decoding bypasses any future initializer-only fix.

### OWN-03

**NormalizationHint retains metadata after direct wrapped mutation or a throwing mutable-buffer closure.**

- Exposure: public wrapper; inspected downstream consumers
- Qualification: The initial hint is true, so this is not caller supplying a false assertion. GPU/downstream end-to-end behavior was not executed; existing contaminated indexes needing rebuild is conditional, not established deployment state.

### OWN-04

**SoAFP16Cache512 treats vector hash digests as exact candidate identity.**

- Exposure: public opt-in cache; no default or sibling caller found
- Qualification: Only the512 cache exists; 384/768/1536 key initializers are currently unused future surface. Full-coordinate hashing alone cannot turn a digest into an equality proof.

### OWN-05

**Compact-tier triangular matrix copies share later mutations.**

- Exposure: public TriangularMatrix default sizes4...32; internal compact counts5...512
- Qualification: Range is derived from m=n(n-1)/2; no claim for an arbitrary configured tier. Iterator snapshot and setAt exposure follow their same stored-copy/setter paths, not independent runtime reproduction.

### OWN-06

**Binary in-place DynamicVector routes omit dimension checks.**

- Exposure: public direct defaults and generic VectorProtocol operator dispatch
- Qualification: Do not extend unchecked dispatch claim to concrete VectorFactory +=, which checks through +. Short RHS optimized overread is static reasoning; no Release overread was intentionally executed.

### OWN-07

**Optimized-vector hashes omit all but first and last eight scalar values.**

- Exposure: all four public optimized vector types; no vector-keyed downstream table found
- Qualification: Legal Hashable collisions are a scaling issue, not equal-values-have-equal-hash violation. O(N^2*d) is worst-case equality work for sufficiently late differences, not measured universal cost. SIMD signed zero is not an additional bug.

### OWN-08

**Pointer normalization documents16-byte alignment as preferred but SIMD loads require it.**

- Exposure: public NormalizeKernels.normalizeUnchecked pointer overload
- Qualification: Report Debug trap and source alignment precondition violation, not a reproduced Release failure. Dimension<4 does not enter SIMD loads. Caller meets documented Float capacity/nonzero magnitude obligations.

### OWN-09

**Vector384Optimized permanently rebinds typed-owner memory where other widths restore binding.**

- Exposure: normal public construction/toArray/buffer access
- Qualification: Keep as binding-lifetime maintenance/contract finding. Layout compatibility is not a blanket strict-aliasing exemption; conversely merely seeing different spellings does not prove an observed miscompile. Alignment of Float-array source must be audited separately across all widths.

### OWN-10

**Compact generic storage does not destroy manually initialized ManagedBuffer tail elements.**

- Exposure: public TriangularMatrix<T>, default compact sizes4...32
- Qualification: Reference-bearing arrays/strings follow missing generic destruction by source; only class sentinel was executed. A ManagedBuffer frees allocation but does not implicitly destroy manually initialized elements.

### OWN-11

**alignedCopy has incompatible initialize versus update semantics across alignment branches.**

- Exposure: public generic utility; no production caller found
- Qualification: This is an unsafe API contract gap, not proof of a fault under a clearly documented supported destination state (none is supplied). Uninitialized-destination crash was not executed. Alignment here means preferred64-byte alignment, not natural-type misalignment.

### OWN-12

**HybridStorage repeating construction/fill performs quadratic buffer traffic.**

- Exposure: DynamicVector repeating/fill/resize above16 Float elements; explicit hybrid heap storage
- Qualification: Exact count1026 belongs to the instrumented fixture; actual Release instruction path independently proves duplicated ownership/copying and quadratic element traffic. Pointer addresses staying unchanged after writes do not refute copying: two replacement allocations can reuse prior address. SIMD stubs are not used to claim numerical results. Generic DimensionStorage default heap mutation does not inherit this finding.

### OWN-13

**TriangularMatrix scalar writes copy complete standard/aligned allocations even when uniquely owned.**

- Exposure: public default matrix n>=33; original emphasized aligned n>=65
- Qualification: Extend original exposure: standard tier513...2048 also detaches on scalar mutation, so repairing only AlignedBuffer.isUniquelyReferenced is insufficient. Address-change evidence uses integer addresses outside borrow scopes, not retained storage copies.

### OWN-14

**Dormant TieredStorage bulk access loses standard writes and inline writes on thrown closure exit.**

- Exposure: internal dormant mutable-buffer route; no production caller found
- Qualification: No current public matrix method routes through this bulk method. It becomes a required dependency if the proposed matrix bulk-fill optimization is introduced.

### OWN-15

**Internal StorageSlice returns scoped pointer without keeping its owner alive.**

- Exposure: internal dormant APIs; no production call found
- Qualification: No supported-use UAF executed. This is distinct from DynamicVector.slice, which returns an owned vector.

### OWN-16

**Experimental consuming into() returns memory still freed by deinit.**

- Exposure: internal compile gate MOVE_ONLY_EXPERIMENTAL absent from package settings
- Qualification: Not compiled in ordinary builds and not runtime-reproduced. Do not enable this gate as an optimization without rederiving all lifetime paths.

### OWN-17

**Public FP16 pointer getters escape ContiguousArray borrow scopes.**

- Exposure: public opt-in fp16Storage getters; kernels use scoped borrows
- Qualification: The getter is explicitly unsafe; do not equate pointer return with a demonstrated supported-use corruption. Retaining the array owner alone does not change the documented nonescape condition of withUnsafeBufferPointer.

### OWN-18

**SoAFP16Cache512 accepts nonpositive maxSize then evicts from an empty list.**

- Exposure: public explicit cache configuration; default100 unaffected
- Qualification: Lower priority configuration boundary. Choose whether zero intentionally disables caching before testing desired policy.

### OWN-19

**FP16 cache caps entry count while payload bytes scale with candidates per entry.**

- Exposure: public opt-in512 cache
- Qualification: Sizing derivation, not attempted allocation/RSS measurement; immutable COW cache values are not themselves an actor data race.

### OWN-20

**StaticDimension.Storage customization is unused by Vector<D> storage yet controls extension availability.**

- Exposure: public custom dimensions
- Qualification: Confusing customization/API gating, not demonstrated numerical failure.

### OWN-21

**VectorTypeFactory optimized wording describes generic fixed-vector selection, not specialized concrete types.**

- Exposure: public factory
- Qualification: No measured performance regression. Changing the output type can alter casts, provider routing and generic overload behavior.

### OWN-22

**VectorCoreOperations creates redundant arrays and temporary results.**

- Exposure: public min/max/clamp operations
- Qualification: Source-visible allocation opportunities; no measured regression or speedup. Constructor cost can additionally be affected by HybridStorage finding.

### OWN-23

**Performance/robustness comments exceed evidence and overlapping storage layers obscure actual ownership.**

- Exposure: public docs and maintainers
- Qualification: General consolidation is an opportunity, not authorization for a wholesale rewrite or proof of universal slowness. Norm arithmetic validation belongs numerical reviewer.

### OWN-24

**SoA/page/raw ownership, size and custom-conformer obligations are already documented.**

- Exposure: unsafe borrowed/transfer/custom conformer surfaces
- Qualification: No additional supported-use corruption established from missing post-transfer checks, @unchecked Sendable, or ordinary checked arithmetic trapping on invalid/oversized shapes. Four built-in layouts do not prove arbitrary custom conformers.

### OWN-25

**AlignedDynamicArrayStorage fallback can weaken alignment while guaranteedAlignment reports the request.**

- Exposure: public allocation-failure fallback
- Qualification: Keep a potential guarantee gap, not reproduced misalignment or corruption.

### OWN-26

**Reject extending HybridStorage quadratic-copy behavior to default generic DimensionStorage without evidence.**

- Exposure: generic Vector<D> default storage; explicit .hybrid separately exposed
- Qualification: Reject blanket claim that linspace/range/generate/geometric/sparse necessarily become quadratic for every generic fixed vector. Explicit .hybrid mode and Double inline threshold deserve dedicated coverage.

### OWN-27

**Extension-only protocol methods select static defaults in generic code.**

- Exposure: public generic VectorProtocol consumers
- Qualification: This alone is not a measured performance defect and does not mean all extensions are wrong. Split narrow read-only protocols only where consumer needs justify change.

### SER-01

**A valid empty DynamicVector binary encoding cannot be decoded by its own type.**

- Exposure: Public binary round trip of default/empty dynamic values.
- Qualification: The 10001 cap is a distinct intentionally tested limitation. Do not remove decoder limits as an incidental repair.

### SER-02

**Serialization tests include 15 comment-only bodies and mislabeled optimized-type coverage.**

- Exposure: Confidence in passing tests.
- Qualification: This does not mean all optimized Codable code is untested elsewhere, or that Core should implement every feature mentioned by a placeholder.

### SER-03

**Binary conversion has scalar loops and temporary representations; a Release allocation speedup is not yet established.**

- Exposure: Binary encode/decode.
- Qualification: Retain only a profiling candidate. Neither one heap allocation per scalar in Release nor a throughput regression was demonstrated; do not blindly replace the implementation.

### SER-04

**Binary and JSON representations have different type/precision contracts.**

- Exposure: Serialization of optimized FP32 and quantized vectors.
- Qualification: No compressed-INT8 binary or universal JSON schema promise found. These are documentation/interoperability decisions, not demonstrated corrupt roundtrips.

### SER-05

**Vectors expose binary methods without adopting the public binary protocols.**

- Exposure: Generic BinaryCodable integrations; no internal dependent caller found.
- Qualification: An integration opportunity; no existing downstream generic BinaryCodable compilation failure was claimed.

### SER-06

**Header documentation describes VECT while the actual little-endian wire bytes spell TCEV.**

- Exposure: External binary readers/documentation.
- Qualification: Existing encoder and decoder agree and tests expect actual bytes. This is a documentation defect; changing wire bytes would be a compatibility change.

### SER-07

**Decoder framing/CRC bounds and Data-slice behavior support the original limited positive assessment.**

- Exposure: Public binary decoder; context/guardrails, not an additional defect.
- Qualification: No public overflow exploit, compression implementation defect or direct-slice bug found. CRC is integrity checking, not authentication. Reserved flags are read but unused; one uncompressed frame is the current contract.

### SER-08

**Decoded INT8 storage, quantization parameters and TopK result arrays require invariant validation.**

- Exposure: Synthesized/custom Codable construction boundaries.
- Qualification: Cross-reference OWN/NUM/API rows for severity and actual effects; this row is boundary coverage, not a separate issue count.

### TOOL-01

**Strict benchmark correctness accepts nonfinite errors, mismatched output lengths and missing validation.**

- Exposure: Benchmark CLI; undermines validation of later optimizations.
- Qualification: NaN JSON failure is separate from threshold acceptance. There was no injected faulty production kernel. This is a validation blocker, not a claim every benchmark value is wrong.

### TOOL-02

**Requested benchmark output can fail to write while the CLI exits successfully.**

- Exposure: Benchmark automation.
- Qualification: The CLI experiment used a filesystem error; NaN-to-CLI exit behavior follows the same catch/control flow.

### TOOL-03

**Benchmark metadata claims Release/C/size settings that do not establish the binary executed.**

- Exposure: Debug and Release CLI metadata; provenance.
- Qualification: Requested flags are not proof a C kernel actually ran. No separately compiled C-on/C-off A/B performance measurement was made.

### TOOL-04

**Extended mixed-precision tests bypass the strict-performance opt-out.**

- Exposure: Manual benchmarks workflow with EXTENDED=1, STRICT_PERF=0.
- Qualification: No current hosted failure or compiler elimination was observed; universal speedup/dimension-order expectations are unsupported by numerical correctness.

### TOOL-05

**Selecting optimized-only dot cases runs zero cases and succeeds.**

- Exposure: Targeted benchmark selection.
- Qualification: This is case selection/coverage loss, not a zero-duration optimized speedup.

### TOOL-06

**Fused/pre-normalized cosine A/B pairs cannot join under their current grouping keys.**

- Exposure: JSON and console A/B reporting.
- Qualification: Euclidean pairing is a separate path and is not invalidated by this proof.

### TOOL-07

**Several timings lack consistent result consumption and compare different setup/allocation boundaries.**

- Exposure: Benchmark/test timing paths.
- Qualification: Missing consumption is proved; dead-code elimination and its timing magnitude are unverified. Different timing scopes can be intentional but must be identified before attributing speedups.

### TOOL-08

**Memory-footprint output counts owning struct handles rather than heap payload.**

- Exposure: Mixed-precision footprint report and misleading timing-test names.
- Qualification: No allocated-byte or RSS measurement was made; payload arithmetic excludes headers, allocator overhead and COW sharing.

### TOOL-09

**Matrix benchmark ignores shared selection/timing controls and has only 17 distinct generated vectors.**

- Exposure: Opt-in matrix benchmark.
- Qualification: No claim that 17 distinct vectors makes numerical computation incorrect; it limits representativeness and provenance.

### TOOL-10

**Current benchmark coverage does not establish downstream large-scale behavior.**

- Exposure: Scale-readiness validation gap; no universal throughput claim supported.
- Qualification: These are coverage opportunities, not proof of a current production outage or a demand to put all workloads into every PR job.

### TOOL-11

**CI coverage excludes important tooling and selected safety surfaces.**

- Exposure: Test/lint/CI configuration.
- Qualification: No remote branch-protection settings inspected; independent checks may be required externally. Declared analyzer rules alone do not execute analysis. Do not count unregistered tests as coverage.

### TOOL-12

**Baseline comparison ignores provenance and CSV metadata does not escape multiline/arbitrary values.**

- Exposure: Benchmark artifact comparison/export.
- Qualification: Historical baselines are useful records, not current guarantees. Aggregate means versus medians are choices needing explicit labeling, not inherently arithmetic errors.

### TOOL-13

**Mobile availability of the separate benchmarking product remains unverified.**

- Exposure: VectorCoreBenchmarking product only.
- Qualification: Do not report a confirmed mobile compile failure or extend this suspicion to the Core product.

