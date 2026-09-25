.PHONY: build-mock sim-mock render-mock view-mock run-all-mock rf-camera-mock rf-camera-calibrate-mock rf-camera-delay-mock rf-camera-multiview-mock rf-camera-optical-mock rf-camera-observe-mock rf-camera-partial-mock rf-gs-toy clean build-mock-city rf-camera-multiview-mock-city rf-camera-coverage-mock rf-camera-coverage-mock-city rf-camera-optical-coverage-mock

# PYTHONPATHを設定
PYTHON := PYTHONPATH=./src/ python

# モックデータのパス
MOCK_JSON := data/raw/mock_building.city.json
MOCK_OUT  := data/generated/mock_results/
MOCK_XML  := $(MOCK_OUT)/mock_building.city.xml
MOCK_MANI := $(MOCK_OUT)/manifest.json
RF_CAMERA_OUT := $(MOCK_OUT)/rf_camera/
RF_CAMERA_MULTIVIEW_OUT := $(MOCK_OUT)/rf_camera_multiview/
RF_CAMERA_PARTIAL_OUT := $(MOCK_OUT)/rf_camera_partial/
MOCK_CITY_JSON := data/raw/mock_city.city.json
MOCK_CITY_OUT  := $(MOCK_OUT)/mock_city/
MOCK_CITY_XML  := $(MOCK_CITY_OUT)/mock_city.city.xml
RF_CAMERA_MULTIVIEW_CITY_OUT := $(MOCK_OUT)/rf_camera_multiview_city/
RF_CAMERA_COVERAGE_OUT := $(MOCK_OUT)/rf_camera_coverage/
RF_CAMERA_COVERAGE_CITY_OUT := $(MOCK_OUT)/rf_camera_coverage_city/
PLACEMENT_SEED ?= 0
ORIENTATION_POLICY ?= face_bs
RADIO_MAP ?=
COVERAGE_RADIO_MAP_ARG := $(if $(RADIO_MAP),--radio-map $(RADIO_MAP),)

build-mock:
	$(PYTHON) -m plateau_rt.cli.main build $(MOCK_JSON) $(MOCK_OUT)

sim-mock:
	$(PYTHON) -m plateau_rt.cli.main simulate $(MOCK_XML) $(MOCK_MANI) $(MOCK_OUT)

render-mock:
	$(PYTHON) -m plateau_rt.cli.main render $(MOCK_OUT)

view-mock:
	$(PYTHON) -m plateau_rt.cli.main view $(MOCK_OUT)

run-all-mock:
	$(PYTHON) -m plateau_rt.cli.main run-all $(MOCK_JSON) $(MOCK_OUT)

rf-camera-mock: build-mock
	$(PYTHON) -m plateau_rt.cli.main rf-camera $(MOCK_XML) $(RF_CAMERA_OUT)

rf-camera-calibrate-mock:
	$(PYTHON) -m plateau_rt.cli.main rf-camera-calibrate $(RF_CAMERA_OUT)

rf-camera-delay-mock:
	$(PYTHON) -m plateau_rt.cli.main rf-camera-delay $(RF_CAMERA_OUT)

rf-camera-multiview-mock: build-mock
	$(PYTHON) -m plateau_rt.cli.main rf-camera-multiview $(MOCK_XML) $(RF_CAMERA_MULTIVIEW_OUT) \
		--num-views 8 --radius-m 30 --ue-height-m 1.5 --target 5 5 5 \
		--bs-position -50 -50 30 --bs-position 60 35 25

rf-camera-optical-mock:
	$(PYTHON) -m plateau_rt.cli.main rf-camera-optical $(RF_CAMERA_MULTIVIEW_OUT)

rf-camera-partial-mock:
	$(PYTHON) -m plateau_rt.cli.main rf-camera-partial $(RF_CAMERA_MULTIVIEW_OUT) $(RF_CAMERA_PARTIAL_OUT) \
		--view-fraction 0.5 --element-mask checkerboard --summary delay --overwrite

build-mock-city:
	$(PYTHON) -m plateau_rt.cli.main build $(MOCK_CITY_JSON) $(MOCK_CITY_OUT) --ground-plane-size-m 200

rf-camera-multiview-mock-city: build-mock-city
	$(PYTHON) -m plateau_rt.cli.main rf-camera-multiview $(MOCK_CITY_XML) $(RF_CAMERA_MULTIVIEW_CITY_OUT) \
		--num-views 12 --radius-m 40 --ue-height-m 1.5 --target 0 0 8 --bs-position -70 5 25

rf-camera-observe-mock:
	$(PYTHON) -m plateau_rt.cli.main rf-camera-observe $(RF_CAMERA_MULTIVIEW_OUT) \
		--front-to-back-db 20 --snr-db 20 --element-gain-std-db 0.5 \
		--element-phase-std-deg 5 --timing-offset-ns 10 --random-common-phase --seed 0

rf-gs-toy:
	$(PYTHON) -m plateau_rt.experimental.rf_scatterer_study --out data/generated/analysis/rf_gs_toy/

rf-camera-coverage-mock: build-mock
	$(PYTHON) -m plateau_rt.cli.main rf-camera-multiview $(MOCK_XML) $(RF_CAMERA_COVERAGE_OUT) \
		--placement coverage --placement-seed $(PLACEMENT_SEED) \
		--orientation-policy $(ORIENTATION_POLICY) $(COVERAGE_RADIO_MAP_ARG) \
		--num-views 8 --ue-height-m 1.5 --target 5 5 5 \
		--bs-position -50 -50 30 --bs-position 60 35 25 \
		--rm-center 0 0 --rm-size 80 80 --rm-cell-size 1 1 \
		--pl-threshold-mode relative_to_max_db --pl-threshold 30 --bs-aggregation any \
		--los-reference all --los-fraction 0.5 \
		--building-clearance-m 1 --min-bs-distance-m 5 --min-ue-spacing-m 5

rf-camera-coverage-mock-city: build-mock-city
	$(PYTHON) -m plateau_rt.cli.main rf-camera-multiview $(MOCK_CITY_XML) $(RF_CAMERA_COVERAGE_CITY_OUT) \
		--placement coverage --placement-seed $(PLACEMENT_SEED) \
		--orientation-policy $(ORIENTATION_POLICY) $(COVERAGE_RADIO_MAP_ARG) \
		--num-views 12 --ue-height-m 1.5 --target 0 0 8 --bs-position -70 5 25 \
		--rm-center 0 0 --rm-size 120 120 --rm-cell-size 1 1 \
		--pl-threshold-mode relative_to_max_db --pl-threshold 50 \
		--building-clearance-m 2 --min-bs-distance-m 10 --min-ue-spacing-m 5

rf-camera-optical-coverage-mock:
	$(PYTHON) -m plateau_rt.cli.main rf-camera-optical $(RF_CAMERA_COVERAGE_OUT)

clean:
	rm -rf data/intermediate/* data/generated/*
