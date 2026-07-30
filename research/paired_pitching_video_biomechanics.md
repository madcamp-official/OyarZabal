# Paired pitching video와 biomechanics 활용 조사

조사일: 2026-07-29

## 결론

공개 데이터 중 `야구 투구 영상 + 정확한 marker-based 3D motion capture + force plate + OBP POI/Full Signal`을 대규모로 함께 제공하는 데이터셋은 확인되지 않았다.

현실적인 조합은 다음과 같다.

1. **SportsPose**의 baseball-pitch subset으로 video-to-3D-pose 모델을 학습·적응한다.
2. **OpenBiomechanics Project(OBP)**의 C3D/landmark/full-signal을 이용해 3D pose-to-POI 변환기를 학습한다.
3. **OBP-CV의 단일 paired throw**로 입출력, 동기화, 좌표계 파이프라인을 검증한다.
4. 최종 정확도는 자체 촬영한 소규모 synchronized multi-view pitching set으로 교정한다.

영상만으로 얻은 GRF, 관절력, 관절 모멘트는 측정값이 아니라 모델 추정값이다. 힘 관련 Full Signal을 ground truth처럼 사용하면 안 된다.

## 공개 후보

| 자료 | 영상과 paired label | 적합한 용도 | 한계 |
|---|---|---|---|
| [SportsPose](https://christianingwersen.github.io/SportsPose/) | 7-view hardware-synchronized 90 Hz RGB, calibration, frame-level 3D joints; baseball pitch 포함 | 투구 특화 2D/3D pose와 event detector 학습 | academic-only 신청 필요, GRF·joint moment·OBP POI 없음 |
| [OBP-CV](https://drive.google.com/drive/folders/1HnLsdim_GfwLmVF_l7wRjPtQ32O4zTUD) | `OBP_movement_BaseballThrow_001` 1개: 8-view 360 fps와 Theia3D C3D/overlay | 데이터 로딩, 동기화, 좌표 변환 smoke test | 단 한 투구, C3D도 markerless pseudo-label, force data 없음, OBP 411 pitches와 join 불가 |
| [Penn Action](https://dreamdragon.github.io/PennAction/) | baseball_pitch 영상, 매 frame 13개 수동 2D joint | 2D pose·투구 구간 검출 사전학습 | 3D, kinetics, 구속, POI 없음 |
| [MLB-YouTube](https://github.com/piergiaj/mlb-youtube) | 4,290 broadcast clips; 투구 clip에 pitch type·speed | broadcast-domain 표현과 구종·구속 보조학습 | biomechanics ground truth 없음; 원영상 권리·가용성 확인 필요 |
| [OpenCap](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462) | multi-view smartphone video에서 3D kinematics/dynamics 추정 파이프라인 | 구현 아키텍처 참고 | pitching으로 검증된 모델이 아니며 힘 값은 physics model 추정 |

추가로 확인한 자료:

- [CMU Panoptic `131015_baseball`](https://domedb.perception.cs.cmu.edu/develop/131015_baseball.html)은 페이지상 영상·calibration·3D pose가 아직 공개되지 않았다.
- [CMU Mocap Subject 124](https://mocap.cs.cmu.edu/search.php?subjectnumber=124)는 baseball pitch mocap은 있지만 paired 원본 RGB 영상은 없다.
- [TU Delft youth pitching dataset](https://data.4tu.nl/datasets/046e600c-4c1b-44b3-afb2-9a1ba015ce9e/1)은 Vicon C3D, landmarks, elbow valgus torque, ball speed를 제공하지만 paired 영상은 없다.
- Hawk-Eye, Theia3D, marker-based mocap을 동시에 비교한 [2025 연구](https://doi.org/10.1080/02640414.2025.2595411)는 이상적인 paired 설계지만 공개 다운로드는 찾지 못했다. 저자에게 데이터 이용 가능성을 문의할 가치가 있다.

## 영상에서 POI와 Full Signal을 만드는 방법

POI는 영상에서 직접 검출하는 별도 센서값이 아니다. 시간축 신호와 이벤트를 먼저 만든 뒤, 특정 시점 값·최댓값·차이·적분을 계산한 요약값이다.

```text
video
  -> camera calibration + synchronization
  -> per-frame 2D keypoints + confidence
  -> multi-view triangulation / monocular 3D lifting
  -> anatomical landmarks or inverse kinematics
  -> filtering and coordinate normalization
  -> foot plant / MER / ball release / MIR event detection
  -> kinematic Full Signal
  -> event values, extrema, differences, integrals
  -> POI
```

[OpenCap](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462)는 이 구조의 좋은 공개 구현 예다. 여러 카메라의 2D keypoint를 동기화·삼각측량하고, sparse keypoint를 anatomical marker set으로 확장한 다음 OpenSim inverse kinematics를 적용한다.

### 영상으로 비교적 타당하게 얻을 수 있는 값

- 2D/3D joint·landmark trajectory
- pelvis, torso, upper arm 등의 orientation
- joint angle과 적절히 필터링한 angular velocity
- foot plant, MER, ball release 같은 event timing
- stride, trunk tilt, arm-slot proxy, hip-shoulder separation
- 위 신호에서 계산한 kinematic POI

다중 시점, 높은 frame rate, camera calibration이 있을수록 정확도가 좋아진다. 방송 단일 영상에서는 깊이, 가림, 줌, camera cut 때문에 절대 3D값보다 정규화된 상대값과 pitcher 내 변화량이 더 안전하다.

### 영상만으로 직접 얻을 수 없는 값

- ground reaction force와 center of pressure
- elbow varus moment, shoulder internal-rotation moment
- joint force, segment torque
- mechanical energy flow

이 값들은 외력과 신체 분절 관성값이 필요한 inverse dynamics 대상이다. Force plate 없이 계산한다면 별도 contact/physics model의 **예측치**이며, pitching-specific paired ground truth로 오차와 calibration을 검증해야 한다.

## OBP와 연결하는 최소 경로

OBP marker topology와 일반 pose-estimation skeleton은 다르므로, 처음부터 81개 POI 전체를 복원하는 것은 권하지 않는다.

1. OBP C3D에서 공통 anatomical landmark와 kinematic Full Signal만 선택한다.
2. OBP 3D landmark를 다양한 가상 카메라로 투영하고 motion blur, occlusion, keypoint noise를 추가한다.
3. noisy 2D/3D pose sequence에서 OBP landmark와 소수 kinematic POI를 예측하도록 학습한다.
4. SportsPose baseball-pitch 영상으로 pose backbone을 domain-adapt한다.
5. 자체 synchronized pitching video+C3D의 작은 bridge set으로 real-video bias를 교정한다.

초기 POI는 다음 5~10개면 충분하다.

- delivery duration / cadence
- foot-plant-to-release duration
- arm-slot proxy
- trunk forward/lateral tilt
- hip-shoulder separation proxy
- stride length proxy
- pelvis/torso angular-velocity proxy
- release-point proxy
- pitcher rolling baseline 대비 mechanics drift

## OyarZabal에 적용할 때의 누수 조건

OyarZabal의 목표가 **다음 투구를 투구 전에 예측**하는 것이라면, 예측 대상 투구의 delivery/release 영상은 사용할 수 없다. 이는 정답 구종이 이미 동작에 나타난 뒤의 정보이므로 target leakage다.

사용 가능한 설계는 이전 투구 `t-1`, `t-2`, ... 영상에서 mechanics feature를 추출해 다음 투구 `t`의 입력으로 lag하는 것이다. 검증도 시간 순서를 지키고 가능하면 pitcher 또는 game 단위 그룹 분할을 병행해야 한다.

우선 권장 실험:

```text
baseline tabular model
vs.
baseline + prior-pitch 5~10 mechanics features
vs.
baseline + prior-pitch pose embedding
```

Full Signal 전체를 재구축하기 전에 저차원 mechanics feature가 out-of-time 성능을 실제로 높이는지 확인하는 편이 가장 저렴하고 검증 가능하다.

## 권리와 운영상 주의

MLB Film Room/Statcast 영상은 소규모 수동 검토 및 human benchmark에는 유용하지만, [MLB 이용약관](https://www.mlb.com/official-information/terms-of-use)은 무단 파생물 제작·재배포와 자동 수집을 제한한다. 대량 frame extraction과 모델 학습에는 별도 허가 또는 권리를 확보한 영상을 사용해야 한다.

따라서 실제 개발 순서는 `SportsPose/OBP-CV로 기술 검증 -> 자체 또는 허가받은 영상으로 bridge set 제작 -> 이전 투구 feature만 OyarZabal에 연결`이 가장 안전하다.
