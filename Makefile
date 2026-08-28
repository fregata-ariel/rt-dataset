.PHONY: build-mock sim-mock render-mock view-mock run-all-mock rf-camera-mock rf-camera-calibrate-mock rf-camera-delay-mock rf-camera-multiview-mock clean

# PYTHONPATHを設定
PYTHON := PYTHONPATH=./src/ python

# モックデータのパス
MOCK_JSON := data/raw/mock_building.city.json
MOCK_OUT  := data/generated/mock_results/
MOCK_XML  := $(MOCK_OUT)/mock_building.city.xml
MOCK_MANI := $(MOCK_OUT)/manifest.json
RF_CAMERA_OUT := $(MOCK_OUT)/rf_camera/
RF_CAMERA_MULTIVIEW_OUT := $(MOCK_OUT)/rf_camera_multiview/

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
	$(PYTHON) -m plateau_rt.adapters.sionna.rf_camera_calibration $(RF_CAMERA_OUT)

rf-camera-delay-mock:
	$(PYTHON) -m plateau_rt.adapters.sionna.rf_camera_delay $(RF_CAMERA_OUT)

rf-camera-multiview-mock: build-mock
	$(PYTHON) -m plateau_rt.adapters.sionna.rf_camera_dataset $(MOCK_XML) $(RF_CAMERA_MULTIVIEW_OUT) \
		--num-views 8 --radius-m 30 --ue-height-m 1.5 --target 5 5 5

clean:
	rm -rf data/intermediate/* data/generated/*
