.PHONY: build-mock sim-mock render-mock view-mock run-all-mock rf-camera-mock clean

# PYTHONPATHを設定
PYTHON := PYTHONPATH=./src/ python

# モックデータのパス
MOCK_JSON := data/raw/mock_building.city.json
MOCK_OUT  := data/generated/mock_results/
MOCK_XML  := $(MOCK_OUT)/mock_building.city.xml
MOCK_MANI := $(MOCK_OUT)/manifest.json
RF_CAMERA_OUT := $(MOCK_OUT)/rf_camera/

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

clean:
	rm -rf data/intermediate/* data/generated/*
