import datetime as dt
import os
import warnings
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests


class ProductML(object):
    def __init__(self, host: str) -> None:

        endpoint = "/api/v1/test"

        print("Testing connection to API ...")

        res = requests.get(host + endpoint)

        if res.status_code == 200:
            print("Successfully connected to API!")
        else:
            raise RuntimeError("Connection failed: " + str(res.text))

        self.host = host

        self.PREDICTION_SCENARIO_OPTIONS = {
            "power_curve": 1,
            "temperature_prediction": 2,
            "solar_performance": 3,
        }  # in the future, get this from the DB!!!
        self.ML_MODEL_TYPE_OPTIONS = {"legacy_kde_regressor": 1}

    def create_client(self, client_id: int, client_name: str, client_settings: Optional[dict]) -> None:

        endpoint = "/api/v1/clients"

        client_settings = {} if not client_settings else client_settings

        input_data = {"id": client_id, "name": client_name, "settings": client_settings}
        res = requests.post(self.host + endpoint, json=input_data)

        if res.status_code == 200:
            print(f"Client {client_name} created!")
        else:
            raise RuntimeError(str(res.text))

    def create_asset(
        self,
        client_id: int,
        asset_id: int,
        asset_name: str,
        asset_type: str,
        parent_asset_id: Optional[int] = None,
        power_curve_id: Optional[int] = 0,
        power_curve_values: Optional[List[List[float]]] = [[0, 0], [0, 0]],  # noqa
    ) -> None:

        endpoint = "/api/v1/assets"

        asset_type_dict = {
            "Wind farm": 1,
            "Wind turbine": 2,
            "Solar farm": 3,
            "Solar panel": 4,
        }  # in the future, get this from DB
        PARENT_ASSETS = {
            "Wind farm",
            "Solar farm",
        }  # in the future, get this from DB (is aggregator = true)

        if asset_type not in asset_type_dict.keys():
            raise ValueError(f"asset_type must be one of {asset_type_dict.keys()}.")

        if asset_type not in PARENT_ASSETS and parent_asset_id is None:
            raise ValueError("parent_asset_id must be specified if new asset is not parent.")

        asset_type_id = asset_type_dict[asset_type]

        input_data = {
            "asset": {
                "name": asset_name,
                "id": asset_id,
                "parent_id": parent_asset_id,
                "client_id": client_id,
                "asset_type_id": asset_type_id,
                "power_curve_id": power_curve_id,
            },
            "power_curve_values": power_curve_values,
        }

        res = requests.post(self.host + endpoint, json=input_data)

        if res.status_code == 200:
            print(f"Asset {asset_name} (client id = {client_id}) created!")
        else:
            raise RuntimeError(str(res.text))

    def get_assets(self, client_id: int, type: str, parent_id: Optional[int] = None) -> pd.DataFrame:

        endpoint = "/api/v1/assets/"

        valid_types = ["child", "aggregator"]
        if type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}.")

        if type == "child" and parent_id is None:
            raise ValueError("parent_id must be specified if asset type is child.")

        input_data = {"client_id": client_id, "limit": 100, "type_which": type}
        if type == "child":
            input_data["parent_id"] = parent_id

        # TODO: figure out why mypy accuses an error in the following line
        res = requests.get(self.host + endpoint, params=input_data)  # type: ignore

        if res.status_code == 200:

            data_lst = [d["Asset"] for d in res.json()["items"]]
            data = pd.DataFrame(
                data_lst,
                columns=["name", "id", "created_at", "updated_at", "asset_type"],
            )
            data["asset_type"] = data["asset_type"].apply(lambda x: x["value"])
            data = data.drop(columns=["created_at", "updated_at"])
            return data

        else:
            raise RuntimeError(str(res.text))

    def _add_data_to_asset(self, asset_id, historical_data):
        endpoint = "/api/v1/attachment"

        if historical_data is None:
            raise ValueError("historical_data must be defined.")

        necessary_columns = [
            "asset_id",
            "read_at",
            "wind_speed",
            "wind_direction",
            "exterior_temperature",
            "power_average",
        ]  # this only works for power curve

        mum_missing_colums = len(list(set(necessary_columns) - set(historical_data.columns.tolist())))

        if mum_missing_colums > 0:
            raise ValueError(f"historical_data must contain columns {necessary_columns}.")

        filename = "asset_data.csv"
        historical_data.to_csv(filename, index=False)

        with open(filename, "rb") as file_to_send:

            files = {"file": file_to_send}

            get_client_res = requests.get(self.host + f"/api/v1/assets/{asset_id}")
            client_id = get_client_res.json()["data"]["client"]["id"]

            input_data = {
                "client_id": client_id,
                "asset_id": asset_id,
                "attachment_type": "asset_data",
            }

            res = requests.post(self.host + endpoint, files=files, params=input_data)

        os.remove(filename)

        if res.status_code == 200:
            print(f"Historical data added to asset id={asset_id} (client id={client_id})!")
            print(
                f"""New attachment created:
                    > id={res.json()['data']['attachment']['id']}
                    > container={res.json()['data']['attachment']['container']}
                    > path={res.json()['data']['attachment']['blob_path']}
                """
            )
            print(
                f"""Processing run activated:
                    > id={res.json()['data']['data_flow_run']['id']}
                    > flow={res.json()['data']['data_flow_run']['data_flow']['id']}
                    > type={res.json()['data']['data_flow_run']['data_flow']['type']['value']}
                """
            )
        else:
            raise RuntimeError(str(res.text))

    def _add_curve_to_asset(self, asset_id, reference_curve):
        endpoint = f"/api/v1/assets/{asset_id}/reference_curve"

        if reference_curve is None:
            raise ValueError("reference_curve must be defined.")

        if "wind_speed" not in reference_curve.columns.tolist() or "power" not in reference_curve.columns.tolist():
            raise ValueError("reference_curve must contain columns wind_speed and power.")

        filename = "curve_data.csv"

        reference_curve.to_csv(filename, index=False)

        files = {"file": open(filename, "rb")}

        res = requests.post(self.host + endpoint, files=files)

        os.remove(filename)

        if res.status_code == 200:
            print(
                f"""Reference curve (id={res.json()['data']['power_curve_id']})
                    was updated for asset id={asset_id} (client id={res.json()['data']['client']['id']})!"""
            )
        else:
            raise RuntimeError(str(res.text))

    def _add_model_to_asset(self, asset_id, ml_model_id, prediction_scenario):
        endpoint = "/api/v1/ml/models"

        if ml_model_id is None:
            raise ValueError("ml_model_id must be defined.")

        if prediction_scenario is None:
            raise ValueError("prediction_scenario must be defined.")

        if prediction_scenario not in self.PREDICTION_SCENARIO_OPTIONS.keys():
            raise ValueError(f"prediction_scenario must be one of {self.PREDICTION_SCENARIO_OPTIONS.keys()}.")

        prediction_scenario_id = self.PREDICTION_SCENARIO_OPTIONS[prediction_scenario]

        input_data = {
            "asset_id": asset_id,
            "model_id": ml_model_id,
            "prediction_scenario_id": prediction_scenario_id,
        }

        res = requests.post(self.host + endpoint, params=input_data)

        if res.status_code == 200:
            print(f"The {prediction_scenario} model (id={ml_model_id}) was added to asset id={asset_id}!")
        else:
            raise RuntimeError(str(res.text))

    def add_to_asset(
        self,
        asset_id: int,
        add: str,
        historical_data: Optional[pd.DataFrame] = None,
        reference_curve: Optional[pd.DataFrame] = None,
        ml_model_id: Optional[int] = None,
        prediction_scenario: Optional[str] = None,
    ) -> None:

        valid_to_add = {"data", "curve", "model"}
        if add not in valid_to_add:
            raise ValueError(f"add must be one of {valid_to_add}.")

        if add == "data":
            self._add_data_to_asset(asset_id, historical_data)
        elif add == "curve":
            self._add_curve_to_asset(asset_id, reference_curve)
        else:
            self._add_model_to_asset(asset_id, ml_model_id, prediction_scenario)

    def list_client_attachments(self, client_id: int) -> pd.DataFrame:

        endpoint = "/api/v1/attachment"
        input_data = {"client_id": client_id}

        res = requests.get(self.host + endpoint, params=input_data)

        if res.status_code == 200:

            attachments_df = pd.DataFrame(res.json()["data"]).rename(columns={"id": "attachment_id"})
            attachments_df = attachments_df[
                ["attachment_id"] + [col for col in attachments_df if col != "attachment_id"]
            ]

            return attachments_df

        else:
            raise RuntimeError(str(res.text))

    def download_attachment(self, attachment_id: int) -> pd.DataFrame:

        endpoint = f"/api/v1/attachment/{attachment_id}/download"
        res = requests.get(self.host + endpoint)

        if res.status_code == 200:
            df = pd.read_csv(StringIO(res.content.decode("utf-8")))
            return df
        else:
            raise RuntimeError(str(res.text))

    def change_attachment(self):
        raise NotImplementedError()

    def delete_attachment(self):
        raise NotImplementedError()

    def _get_asset_data_without_predictions(self, asset_id, start_date, end_date, features_to_include, endpoint):
        input_data: Dict[str, Any] = {
            "start_date": start_date,
            "final_date": end_date,
        }

        res = requests.get(self.host + endpoint, params=input_data)

        if res.status_code == 200:

            df = pd.DataFrame(res.json()["items"])
            df["asset_id"] = asset_id

            if features_to_include == "all":
                pass
            else:
                if isinstance(features_to_include, str):
                    features_to_include = [features_to_include]
                elif features_to_include is None:
                    features_to_include = []
                df = df[["asset_id", "read_at", "power_average"] + features_to_include].copy()
            df["read_at"] = pd.to_datetime(df["read_at"])

            return df

        else:
            raise RuntimeError(str(res.text))

    def _get_asset_data_with_predictions(
        self,
        asset_id,
        start_date,
        end_date,
        features_to_include,
        prediction_scenario,
        endpoint,
    ):
        if prediction_scenario not in list(self.PREDICTION_SCENARIO_OPTIONS.keys()) + ["all"]:
            raise ValueError(f"prediction_scenario must be one of {self.PREDICTION_SCENARIO_OPTIONS.keys()}.")

        if prediction_scenario == "all":
            prediction_scenarios_list = list(self.PREDICTION_SCENARIO_OPTIONS.keys())
        else:
            prediction_scenarios_list = [prediction_scenario]

        data = []

        for prediction_scenario_item in prediction_scenarios_list:

            prediction_scenario_id = self.PREDICTION_SCENARIO_OPTIONS[prediction_scenario_item]

            input_data = {
                "start_date": start_date,
                "final_date": end_date,
                "prediction_scenario_id": prediction_scenario_id,
            }

            res = requests.get(self.host + endpoint, params=input_data)

            if res.status_code == 200:
                df = pd.DataFrame(res.json()["items"])
                if len(df) == 0:
                    raise RuntimeError("Returned empty dataframe.")
                df["asset_id"] = asset_id

                if features_to_include == "all":
                    pass
                else:
                    if isinstance(features_to_include, str):
                        features_to_include = [features_to_include]
                    elif features_to_include is None:
                        features_to_include = []
                    df = df[["asset_id", "read_at", "power_average", "expected_power"] + features_to_include].copy()
                df["prediction_scenario"] = prediction_scenario_item
                df["read_at"] = pd.to_datetime(df["read_at"])

                data.append(df)
            else:
                warnings.warn(f"Warning: Bad response for prediction_scenario {input_data} - {res.text}")

        data = pd.concat(data, ignore_index=True)

        return data

    def get_asset_data(
        self,
        asset_id: int,
        start_date: str,
        end_date: str,
        include_predictions: bool = False,
        prediction_scenario: Optional[str] = None,
        features_to_include: Optional[Union[List[str], str]] = "all",
    ) -> pd.DataFrame:

        endpoint = f"/api/v1/assets/{asset_id}/data"

        try:
            dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD HH:MM:SS")

        if not include_predictions:
            return self._get_asset_data_without_predictions(
                asset_id, start_date, end_date, features_to_include, endpoint
            )
        else:
            return self._get_asset_data_with_predictions(
                asset_id,
                start_date,
                end_date,
                features_to_include,
                prediction_scenario,
                endpoint,
            )

    def list_asset_models(
        self,
        client_id: int,
        asset_id: int,
        ml_model_type: str,
        prediction_scenario: str,
    ) -> pd.DataFrame:

        endpoint = "/api/v1/ml/models"

        if ml_model_type not in list(self.ML_MODEL_TYPE_OPTIONS.keys()) + ["all"]:
            raise ValueError(f"ml_model_type must be one of {self.ML_MODEL_TYPE_OPTIONS.keys()}.")

        if prediction_scenario not in list(self.PREDICTION_SCENARIO_OPTIONS.keys()) + ["all"]:
            raise ValueError(f"prediction_scenario must be one of {self.PREDICTION_SCENARIO_OPTIONS.keys()}.")

        if ml_model_type == "all":
            model_types_list = list(self.ML_MODEL_TYPE_OPTIONS.keys())
        else:
            model_types_list = [ml_model_type]

        if prediction_scenario == "all":
            prediction_scenarios_list = list(self.PREDICTION_SCENARIO_OPTIONS.keys())
        else:
            prediction_scenarios_list = [prediction_scenario]

        models_df_lst = []

        for prediction_scenario_item in prediction_scenarios_list:

            for model_type_item in model_types_list:

                ml_model_type_id = self.ML_MODEL_TYPE_OPTIONS[model_type_item]
                prediction_scenario_id = self.PREDICTION_SCENARIO_OPTIONS[prediction_scenario_item]

                input_data = {
                    "client_id": client_id,
                    "asset_id": asset_id,
                    "model_type_id": ml_model_type_id,
                    "prediction_scenario_id": prediction_scenario_id,
                }

                res = requests.get(self.host + endpoint, params=input_data)

                if res.status_code == 200:

                    df = pd.DataFrame(res.json()["items"]).rename(columns={"id": "model_id"})
                    models_df_lst.append(df)

                else:
                    warnings.warn(f"Warning: Bad response for input data {input_data} - {res.text}")

        models_df = pd.concat(models_df_lst, ignore_index=True)

        if len(models_df) > 0:

            models_df["model_type"] = models_df["model_type"].apply(lambda x: x["value"])
            models_df["prediction_scenario"] = models_df["prediction_scenario"].apply(lambda x: x["value"])
            models_df["state"] = models_df["state"].apply(lambda x: x["value"])

            # TODO is the list comprehension not supposed to iterate the dataframe's columns ?
            # If it is, isn't it redundant to add 'model_id' back?
            # Proposed line:
            # models_df = models_df[['model_id'] + [col for col in models_df.columns if col != 'model_id']]
            models_df = models_df[["model_id"] + [col for col in models_df if col != "model_id"]]

            return models_df

        else:
            raise RuntimeError("No data for specified input!")

    def train(
        self,
        asset_id: int,
        client_id: str,
        ml_model_type: str,
        prediction_scenario: str,
        start_date: str,
        end_date: str,
    ) -> None:

        endpoint = "/api/v1/ml/train"

        if ml_model_type not in self.ML_MODEL_TYPE_OPTIONS.keys():
            raise ValueError(f"ml_model_type must be one of {self.ML_MODEL_TYPE_OPTIONS.keys()}.")

        ml_model_type_id = self.ML_MODEL_TYPE_OPTIONS[ml_model_type]

        if prediction_scenario not in self.PREDICTION_SCENARIO_OPTIONS.keys():
            raise ValueError(f"prediction_scenario must be one of {self.PREDICTION_SCENARIO_OPTIONS.keys()}.")

        prediction_scenario_id = self.PREDICTION_SCENARIO_OPTIONS[prediction_scenario]

        try:
            dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD HH:MM:SS")

        input_data = {
            "asset_id": asset_id,
            "ml_model_type_id": ml_model_type_id,
            "prediction_scenario_id": prediction_scenario_id,
            "client_id": client_id,
            "start_date": start_date,
            "final_date": end_date,
        }

        res = requests.post(self.host + endpoint, json=input_data)

        if res.status_code == 200:
            print(
                f"""Training process of {ml_model_type} started for asset id={asset_id} (client id={client_id})!
                    Processing run id: {res.json()['data'][0]['id']}"""
            )
        else:
            raise RuntimeError(str(res.text))

    def predict(
        self,
        asset_id: int,
        client_id: str,
        ml_model_id: str,
        start_date: str,
        end_date: str,
    ) -> None:

        endpoint = "/api/v1/ml/inference"

        try:
            dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
            dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD HH:MM:SS")

        input_data = {
            "asset_id": asset_id,
            "client_id": client_id,
            "ml_model_id": ml_model_id,
            "start_date": start_date,
            "final_date": end_date,
        }

        res = requests.post(self.host + endpoint, json=input_data)

        if res.status_code == 200:
            print(
                f"""Prediction process of model id={ml_model_id}
                    started for asset id={asset_id} (client id={client_id})!
                    Processing run id: {res.json()['data'][0]['id']}"""
            )
        else:
            raise RuntimeError(str(res.text))

    def get_process(self):
        raise NotImplementedError()

    def get_asset_processes(self, asset_id: int, num_records: Optional[int] = 50) -> pd.DataFrame:

        endpoint = "/api/v1/data_flows/runs/"

        input_data = {"asset_id": asset_id, "limit": num_records}

        res = requests.get(self.host + endpoint, params=input_data)

        if res.status_code == 200:

            df = pd.DataFrame(res.json()["items"])

            try:

                runs_data_lst = []
                for run_id in df["id"].unique():

                    endpoint = f"/api/v1/data_flows/runs/{run_id}"

                    res = requests.get(self.host + endpoint)
                    runs_data_lst.append(res.json()["data"])

                runs_data = pd.DataFrame(runs_data_lst)

                runs_data = runs_data.drop(columns=["prefect_ref_id", "data_flow", "meta_data"])
                runs_data["state"] = runs_data["state"].apply(lambda x: x["value"])

                endpoint = "/api/v1/data_flows/"

                res = requests.get(self.host + endpoint)
                flows_df = pd.DataFrame(res.json()["data"])
                flows_df = flows_df.rename(columns={"id": "data_flow_id", "type": "data_flow_type"}).drop(
                    columns=["prefect_ref_id", "created_at", "updated_at"]
                )
                flows_df["data_flow_type"] = flows_df["data_flow_type"].apply(lambda x: x["value"])

                final_df = runs_data.merge(flows_df, on="data_flow_id")

                final_df["created_at"] = pd.to_datetime(final_df["created_at"])
                final_df["updated_at"] = pd.to_datetime(final_df["updated_at"])

                final_df["duration"] = final_df.apply(
                    lambda row: (row["updated_at"] - row["created_at"]).total_seconds(),
                    axis=1,
                )

                final_df = final_df.drop(columns=["updated_at"]).rename(columns={"id": "run_id"})
                final_df = final_df[
                    [
                        "run_id",
                        "asset_id",
                        "created_at",
                        "state",
                        "duration",
                        "error_message",
                        "kwargs",
                        "data_flow_id",
                        "data_flow_type",
                    ]
                ]

                return final_df

            except Exception as error:
                raise RuntimeError(error)

        else:
            raise RuntimeError(str(res.text))

    def show_clients(self):
        raise NotImplementedError()

    def client_report(self):
        raise NotImplementedError()
