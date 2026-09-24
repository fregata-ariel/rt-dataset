.PHONY: build-mock sim-mock render-mock view-mock run-all-mock rf-camera-mock rf-camera-calibrate-mock rf-camera-delay-mock rf-camera-multiview-mock rf-camera-optical-mock clean build-mock-city rf-camera-multiview-mock-city

# PYTHONPATHを設定
PYTHON := PYTHONPATH=./src/ python

# モックデータのパス
MOCK_JSON := data/raw/mock_building.city.json
MOCK_OUT  := data/generated/mock_results/
MOCK_XML  := $(MOCK_OUT)/mock_building.city.xml
MOCK_MANI := $(MOCK_OUT)/manifest.json
RF_CAMERA_OUT := $(MOCK_OUT)/rf_camera/
RF_CAMERA_MULTIVIEW_OUT := $(MOCK_OUT)/rf_camera_multiview/
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

build-mock-city:
	$(PYTHON) -m plateau_rt.cli.main build $(MOCK_CITY_JSON) $(MOCK_CITY_OUT) --ground-plane-size-m 200

rf-camera-multiview-mock-city: build-mock-city
	$(PYTHON) -m plateau_rt.cli.main rf-camera-multiview $(MOCK_CITY_XML) $(RF_CAMERA_MULTIVIEW_CITY_OUT) \
		--num-views 12 --radius-m 40 --ue-height-m 1.5 --target 0 0 8 --bs-position -70 5 25

clean:
	rm -rf data/intermediate/* data/generated/*
