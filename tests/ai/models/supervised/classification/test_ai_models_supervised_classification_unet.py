import pytest


def test_u_net_model_evaluate():
	from joatmon.ai.models.supervised.classification.unet import UNetModel

	model = UNetModel(channels=3, classes=100)
	model.evaluate()

	assert True is True


def test_u_net_model_initialize():
	from joatmon.ai.models.supervised.classification.unet import UNetModel

	model = UNetModel(channels=3, classes=100)
	model.initialize()

	assert True is True


def test_u_net_model_load():
	from joatmon.ai.models.supervised.classification.unet import UNetModel

	model = UNetModel(channels=3, classes=100)
	model.save('weights/unet')
	model.load('weights/unet')

	assert True is True


def test_u_net_model_predict():
	from joatmon.ai.models.supervised.classification.unet import UNetModel

	model = UNetModel(channels=3, classes=100)
	model.predict()

	assert True is True


def test_u_net_model_save():
	from joatmon.ai.models.supervised.classification.unet import UNetModel

	model = UNetModel(channels=3, classes=100)
	model.save('weights/unet')

	assert True is True


def test_u_net_model_train():
	from joatmon.ai.models.supervised.classification.unet import UNetModel

	model = UNetModel(channels=3, classes=100)
	model.train()

	assert True is True


if __name__ == '__main__':
	pytest.main([__file__])
