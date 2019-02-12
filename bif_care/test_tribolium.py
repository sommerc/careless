import os
from utils import JVM
from core import BifCareInputConverter, BifCareTrainer

class TestCaseTribolium(object):
    config = {
        "in_dir" : "test_tribolium/in",
        "out_dir": "test_tribolium/out",
        "low_wc" : "*low*",
        "high_wc" : "*high*",
        "patch_size": [16, 128, 128], 
        "n_patches_per_image": 128, 
        "train_epochs": 2, 
        "train_steps_per_epoch": 10, 
        "train_batch_size": 16, 
        "train_channels": [0], 
        "low_scaling": [1.0, 1.0, 1.0]
    }
    predict_fn = "nGFP_0.1_0.2_0.5_20_14_late_low.tif"
    
    def test_convert(self):
        bifc = BifCareInputConverter(**self.config)
        bifc.convert()

    def test_create_patches(self):
        bift = BifCareTrainer(**self.config)
        bift.create_patches()

    def test_train(self):
        bift = BifCareTrainer(**self.config)
        bift.train()
        bift.save()

    def test_predict(self):
        bift = BifCareTrainer(**self.config)
        bift.predict(os.path.join(TestCaseTribolium.config["in_dir"], self.predict_fn))


if __name__ == "__main__":
    JVM().start()
    try:
        tt = TestCaseTribolium()
        tt.test_convert()
        tt.test_create_patches()
        tt.test_train()
        tt.test_predict()
        print("If you read this, tests have passed...")
    except Exception as e:
        raise
    finally:
        JVM().shutdown()