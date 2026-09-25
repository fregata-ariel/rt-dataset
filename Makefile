.PHONY: build-mock sim-mock render-mock view-mock run-all-mock rf-camera-mock rf-camera-calibrate-mock rf-camera-delay-mock rf-camera-multiview-mock rf-camera-optical-mock rf-camera-observe-mock rf-camera-partial-mock rf-gs-toy clean build-mock-city rf-camera-multiview-mock-city viewer-up viewer-down

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

clean:
	rm -rf data/intermediate/* data/generated/*

# viewer (docker compose, GPU 不要): http://127.0.0.1:$(VIEWER_PORT)/
VIEWER_PORT ?= 8765
viewer-up:
	mkdir -p data/viewer data/generated
	VIEWER_PORT=$(VIEWER_PORT) docker compose up -d --build --wait viewer
	@echo "viewer: http://127.0.0.1:$(VIEWER_PORT)/"

viewer-down:
	VIEWER_PORT=$(VIEWER_PORT) docker compose down viewer
