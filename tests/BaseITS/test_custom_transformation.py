import pytest
import pandas as pd

from BaseITS.custom_transform import CustomTransform


@pytest.fixture(scope="class")
def custom_transform():
    return CustomTransform(["pandemic_april", "time"])


@pytest.mark.usefixtures("custom_transform")
class Test_CustomTransform:
    def test_fit(self, X, y, poisson_df, custom_transform):
        """Nothing to test here, no function definition

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        assert True

    def test_transform(self, poisson_df, custom_transform):
        value = custom_transform.transform(poisson_df)

        assert isinstance(value, pd.DataFrame) and {
            "pandemic_april",
            "time",
            "harmonic(month,2,12) 1",
            "harmonic(month,2,12) 2",
            "harmonic(month,2,12) 3",
            "harmonic(month,2,12) 4",
        }.issubset(set(value.columns.tolist()))
