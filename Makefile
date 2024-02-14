# Install dependencies
install:
	@./setup.sh install

# Run the UI
run:
	python quick_draw.py $(MODEL_PATH) $(CONFIG_PATH)

train:
	python train.py $(CONFIG_PATH)

# Delete venv
clean:
	@./setup.sh clean

.PHONY: install run train clean
