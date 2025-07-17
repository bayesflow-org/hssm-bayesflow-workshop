import ssms

from ssms.basic_simulators.simulator import simulator
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
from pathlib import Path


class SSMDataGenerator:
    """A class to generate data from Sequential Sampling Models (SSM).

    This class handles the generation of simulated data from various sequential sampling models,
    including parameter sampling and trial generation across multiple subjects.

    Parameters
    ----------
    model : str
        The name of the sequential sampling model to use
    n_participants : int
        Number of subjects to simulate
    n_trials : int
        Number of trials per subject

    Attributes
    ----------
    model : str
        The name of the sequential sampling model
    model_config : dict
        Configuration dictionary for the specified model
    model_params : list
        List of parameter names for the model
    n_participants : int
        Number of subjects
    n_trials : int
        Number of trials per subject
    param_buffer : float
        Buffer to constrain parameters away from bounds
    param_std_range_fraction : float
        Fraction of parameter range to use for standard deviation
    df : pandas.DataFrame or None
        DataFrame containing generated data
    group_params : dict
        Dictionary of group-level parameters
    subject_params : dict
        Dictionary of subject-level parameters
    state : dict
        Dictionary tracking simulation state
    """

    def __init__(self,
                 model,
                 n_participants: int,
                 n_trials: int) -> None:

        self.model = model
        self.model_config = deepcopy(ssms.config.model_config[model])
        self.model_params = self.model_config["params"]

        self.n_participants = n_participants
        self.n_trials = n_trials

        # Parameter constraints
        self.param_buffer = 0.05
        self.param_std_range_fraction = 1 / 9
        self.continuous_regression_beta_mean_bounds = [-0.5, 0.5]

        # Supplied parameters
        self.group_params_supplied = {}
        self.subject_params_supplied = {}

        # Data
        self.df = None
        self.group_params = {}
        self.subject_params = {}
        self.state = {"sampled": False,
                      "injected_sequential_dynamics": False}

        # RL Rules
        self.basic_rl_config = {"learnable_model_param": "v",
                                "params": ["rl_scaler", "rl_alpha"],
                                "param_bounds": [[2.0, 0.0], [5.0, 1.0]],
                                }
        self.basic_rl_params = self.basic_rl_config["params"]
        self.basic_rl_env = {"ch0_mean": 0.2,
                             "ch0_std": 0.1,
                             "ch1_mean": 0.8,
                             "ch1_std": 0.1
                             }

    def get_parameter_constraints(self,
                                  target_parameter: str,
                                  model_config: dict | None = None,
                                  param_buffer: float | None = None) -> tuple[list[float], float, float]:
        """Get parameter constraints for a given model parameter.

        Parameters
        ----------
        target_parameter : str
            Name of the parameter to get constraints for

        Returns
        -------
        param_bounds_buffered : list[float]
            Lower and upper bounds for the parameter with buffer applied
        param_bounds_range : float
            Range between buffered bounds
        param_bounds_std_max : float
            Maximum allowed standard deviation for the parameter
        """

        if model_config is None:
            model_config = self.model_config
        if param_buffer is None:
            param_buffer = self.param_buffer

        param_bounds = [model_config["param_bounds"][0][model_config["params"].index(target_parameter)],
                        model_config["param_bounds"][1][model_config["params"].index(target_parameter)]]

        param_bounds_buffered = [0] * 2
        param_bounds_buffered[0] = param_bounds[0] + param_buffer
        param_bounds_buffered[1] = param_bounds[1] - param_buffer
        param_bounds_range = param_bounds_buffered[1] - param_bounds_buffered[0]
        param_bounds_std_max = param_bounds_range * self.param_std_range_fraction

        return param_bounds_buffered, param_bounds_range, param_bounds_std_max

    def get_parameter_constraints_vec(self,
                                      model_config: dict | None = None,
                                      param_buffer: float | None = None) -> tuple[list[float], np.ndarray, np.ndarray]:
        """Get parameter constraints vectors for all parameters of a given model.

        Returns
        -------
        param_bounds_buffered : list[float]
            Lower and upper bounds for all parameters with buffer applied
        param_bounds_range : numpy.ndarray
            Range between buffered bounds for all parameters
        param_bounds_std_max : numpy.ndarray
            Maximum allowed standard deviation for all parameters
        """

        if model_config is None:
            model_config = self.model_config
        if param_buffer is None:
            param_buffer = self.param_buffer

        param_bounds = model_config["param_bounds"]
        param_bounds_buffered = deepcopy(param_bounds)
        param_bounds_buffered[0] = np.array(param_bounds)[0] + param_buffer
        param_bounds_buffered[1] = np.array(param_bounds)[1] - param_buffer
        param_bounds_range = param_bounds_buffered[1] - param_bounds_buffered[0]
        param_bounds_std_max = param_bounds_range * self.param_std_range_fraction

        return param_bounds_buffered, param_bounds_range, param_bounds_std_max

    def prep_dfs_for_saving(self,
                            first_columns: list = ["response", "rt", "participant_id", "trial"]) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """Order columns of DataFrames to match the HSSM standard format and create modeling subset.

        This function reorders the columns of a DataFrame to place specified columns at the beginning,
        followed by parameters, and then by model-specific parameters. It also creates a subset
        DataFrame containing only the columns needed for modeling.

        Parameters
        ----------
        first_columns : list, optional
            List of column names to place at the beginning of the DataFrame, by default ["response", "rt", "participant_id", "trial"]

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            - DataFrame with reordered columns including all parameters
            - DataFrame subset containing only modeling-relevant columns

        Raises
        ------
        ValueError
            If attempting to get modeling df before sampling data
        """

        if not self.state["sampled"]:
            raise ValueError("Attempt to get modeling df before sampling data")

        df_tmp = deepcopy(self.df)
        columns_ = list(df_tmp.columns)
        first_cols_ = [col for col in first_columns if col in columns_]
        remain_cols_ = [col for col in columns_ if col not in first_cols_]
        columns_hssm = first_cols_ + remain_cols_
        df_ordered = df_tmp[columns_hssm]

        modeling_cols = [col for col in df_ordered.columns if \
                         (("beta" not in col) and (col not in self.model_params)) and ("rl_" not in col)]
        df_modeling = df_ordered[modeling_cols]

        return df_ordered, df_modeling

    def inject_trials(self) -> pd.DataFrame:
        """Inject trials into the dataframe by repeating each subject's data n_trials times.

        Returns
        -------
        pd.DataFrame
            DataFrame with injected trials, where each subject's data is repeated n_trials times
            and a trial counter is added
        """
        df_tmp = deepcopy(self.df)
        df_repeated = df_tmp.loc[np.repeat(df_tmp.index,
                                           self.n_trials)]. \
            reset_index(drop=True)
        df_repeated["trial"] = df_repeated.groupby("participant_id").cumcount() + 1

        print("Adding trials to attached df")
        self.df = df_repeated
        return df_repeated

    def attach_simulations_to_df(self) -> pd.DataFrame:
        """Attach simulated response times and choices to the dataframe.

        Returns
        -------
        pd.DataFrame
            DataFrame with simulated response times and choices attached

        Raises
        ------
        ValueError
            If attempting to resample when sequential dynamics have been injected
        """
        if self.state["sampled"] and self.state["injected_sequential_dynamics"]:
            raise ValueError(
                "Disallowing resampling because sequential dynamics (e.g. past dependence) have been injected"
                "This is to ensure that the injected dynamics are not overriden by the resampling.")

        df_tmp = deepcopy(self.df)
        sim_out = simulator(model=self.model, theta=df_tmp[self.model_params], n_samples=1)
        df_tmp["rt"] = sim_out["rts"].squeeze()
        df_tmp["response"] = sim_out["choices"].squeeze()

        print("Attaching simulations to df")
        self.df = df_tmp
        self.state["sampled"] = True
        return df_tmp

    def get_parameter_sets_hierarchical(self,
                                        include_basic_rl: bool = False) -> tuple[pd.DataFrame, dict, dict]:
        """Generate hierarchical parameter sets for a given model.

        Parameters
        ----------
        include_basic_rl : bool, optional
            Whether to include basic RL parameters, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame containing subject-level parameters and group-level parameters
        dict
            Dictionary containing group-level parameters
        dict
            Dictionary containing subject-level parameters

        Raises
        ------
        ValueError
            If unable to sample parameters within bounds after 100 attempts
        """
        dfs = []
        group_params = {}
        subject_params = {}
        for param in self.model_params:
            param_bounds_buffered, param_bounds_range, param_bounds_std_max = \
                self.get_parameter_constraints(param)
            param_bounds_mean = (param_bounds_buffered[1] + param_bounds_buffered[0]) / 2
            max_mean_perturbation = param_bounds_std_max * 2

            # Sample subject level parameters
            cnt = 0
            while True:
                if f"beta_{param}_intercept_mean" in self.group_params_supplied:
                    group_sample_mean = self.group_params_supplied[f"beta_{param}_intercept_mean"]
                else:
                    # Sample mean perturbation
                    param_bounds_mean_perturbation = np.random.uniform(-max_mean_perturbation, max_mean_perturbation)
                    group_sample_mean = param_bounds_mean + param_bounds_mean_perturbation

                if f"beta_{param}_intercept_std" in self.group_params_supplied:
                    group_sample_std = self.group_params_supplied[f"beta_{param}_intercept_std"]
                else:
                    # Sample std perturbation
                    group_sample_std = np.random.uniform(0.05, np.maximum(0.05, param_bounds_std_max))

                # Subject level parameters
                if f"beta_{param}_intercept_subject" in self.subject_params_supplied:
                    subject_level_parameters = self.subject_params_supplied[f"beta_{param}_intercept_subject"]
                else:
                    subject_level_parameters = np.random.normal(loc=group_sample_mean,
                                                                scale=group_sample_std,
                                                                size=(self.n_participants))

                # check if all parameters are within the bounds
                if np.all(subject_level_parameters < (param_bounds_buffered[1])) and \
                        np.all(subject_level_parameters > (param_bounds_buffered[0])):
                    break
                elif cnt > 100:
                    raise ValueError("Failed to sample parameters within bounds after 100 attempts")
                cnt += 1

            df_tmp = pd.DataFrame(subject_level_parameters, columns=[param])
            df_tmp[f"beta_{param}_intercept_mean"] = group_sample_mean
            df_tmp[f"beta_{param}_intercept_std"] = group_sample_std
            df_tmp[f"beta_{param}_intercept_subject"] = subject_level_parameters

            # Update parameter_dicts
            group_params.update({
                f"beta_{param}_intercept_mean": group_sample_mean,
                f"beta_{param}_intercept_std": group_sample_std,
            })
            subject_params.update({
                f"beta_{param}_intercept_subject": subject_level_parameters,
            })
            dfs.append(df_tmp)

        if include_basic_rl:
            for param in self.basic_rl_params:
                param_bounds_buffered, param_bounds_range, param_bounds_std_max = \
                    self.get_parameter_constraints(param,
                                                   model_config=self.basic_rl_config)
                param_bounds_mean = (param_bounds_buffered[1] + param_bounds_buffered[0]) / 2
                max_mean_perturbation = param_bounds_std_max * 2

                # Sample subject level parameters
                cnt = 0
                while True:
                    if f"{param}_group_mean" in self.group_params_supplied:
                        group_sample_mean = self.group_params_supplied[f"{param}_group_mean"]
                    else:
                        # Sample mean perturbation
                        param_bounds_mean_perturbation = np.random.uniform(-max_mean_perturbation,
                                                                           max_mean_perturbation)
                        group_sample_mean = param_bounds_mean + param_bounds_mean_perturbation

                    if f"{param}_group_std" in self.group_params_supplied:
                        group_sample_std = self.group_params_supplied[f"{param}_group_std"]
                    else:
                        # Sample std perturbation
                        group_sample_std = np.random.uniform(0.05, np.maximum(0.05, param_bounds_std_max))

                    # Subject level parameters
                    if f"{param}_subject" in self.subject_params_supplied:
                        subject_level_parameters = self.subject_params_supplied[f"{param}_subject"]
                    else:
                        subject_level_parameters = np.random.normal(loc=group_sample_mean,
                                                                    scale=group_sample_std,
                                                                    size=(self.n_participants))

                    # check if all parameters are within the bounds
                    if np.all(subject_level_parameters < (param_bounds_buffered[1])) and \
                            np.all(subject_level_parameters > (param_bounds_buffered[0])):
                        break
                    elif cnt > 100:
                        raise ValueError("Failed to sample parameters within bounds after 100 attempts")
                    cnt += 1

                df_tmp = pd.DataFrame(subject_level_parameters, columns=[param])
                df_tmp[f"{param}_group_mean"] = group_sample_mean
                df_tmp[f"{param}_group_std"] = group_sample_std
                df_tmp[f"{param}_subject"] = subject_level_parameters

                # Update parameter_dicts
                group_params.update({
                    f"{param}_group_mean": group_sample_mean,
                    f"{param}_group_std": group_sample_std,
                })
                subject_params.update({
                    f"{param}_subject": subject_level_parameters,
                })

                dfs.append(df_tmp)

        df = pd.concat(dfs, axis=1)
        df = df[self.model_params + df.columns.difference(self.model_params).tolist()]
        df["participant_id"] = np.arange(self.n_participants).astype(str)
        print("Attaching subject level parameters")
        self.df = df

        # Delete the parameter that is learned via rl from parameter dictionaries
        del group_params[f"beta_{self.basic_rl_config['learnable_model_param']}_intercept_mean"]
        del group_params[f"beta_{self.basic_rl_config['learnable_model_param']}_intercept_std"]
        del subject_params[f"beta_{self.basic_rl_config['learnable_model_param']}_intercept_subject"]

        # Update parameter dictionaries
        self.group_params.update(group_params)
        self.subject_params.update(subject_params)
        return self.df, self.group_params, self.subject_params

    def inject_categorical_regression(self,
                                      target_parameter: str,
                                      covariate_name: str = "x",
                                      group_beta_sign: str | None = "positive") -> tuple[pd.DataFrame, dict, dict]:
        """Injects categorical regression effects into parameter values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing subject and trial data
        model : str
            Name of the model being used
        target_parameter : str
            Parameter to inject categorical effects into
        categorical_name : str, optional
            Name of the categorical variable, by default "x"

        Returns
        -------
        tuple[pd.DataFrame, dict, dict]
            Modified DataFrame, group parameters dict, and subject parameters dict

        Raises
        ------
        ValueError
            If unable to sample parameters within bounds after 100 attempts
        """
        param_bounds_buffered, param_bounds_range, param_bounds_std_max = self.get_parameter_constraints(
            target_parameter)
        n_participants = len(self.df["participant_id"].unique())
        n_trials = len(self.df["trial"].unique())

        # Inject categorical covariate
        group_params = {}
        subject_params = {}

        cnt = 0
        while True:
            df_tmp = deepcopy(self.df)
            # Add categorical column to dataframe
            df_tmp[f"{covariate_name}"] = np.random.choice([0, 1],
                                                           size=(n_participants * n_trials),
                                                           p=[0.5, 0.5])

            # Sample group level parameters
            if f"beta_{target_parameter}_{covariate_name}_group_mean" in self.group_params_supplied:
                beta_group_mean = self.group_params_supplied[f"beta_{target_parameter}_{covariate_name}_group_mean"]
            else:
                if group_beta_sign == "positive":
                    beta_group_mean = np.random.uniform(0.1, param_bounds_range * 1 / 5)
                elif group_beta_sign == "negative":
                    beta_group_mean = np.random.uniform(-(param_bounds_range * 1 / 5), -0.1)
                else:
                    beta_group_mean = np.random.uniform(-(param_bounds_range * 1 / 5),
                                                        (param_bounds_range * 1 / 5))

            if f"beta_{target_parameter}_{covariate_name}_group_std" in self.group_params_supplied:
                beta_group_std = self.group_params_supplied[f"beta_{target_parameter}_{covariate_name}_group_std"]
            else:
                beta_group_std = np.random.uniform(0.05, np.maximum(0.05, param_bounds_std_max))

            if f"beta_{target_parameter}_{covariate_name}_subject" in self.subject_params_supplied:
                beta_subject = self.subject_params_supplied[f"beta_{target_parameter}_{covariate_name}_subject"]
            else:
                beta_subject = np.random.normal(beta_group_mean, beta_group_std, size=n_participants)

            df_tmp[f"beta_{target_parameter}_{covariate_name}_group_mean"] = \
                beta_group_mean
            df_tmp[f"beta_{target_parameter}_{covariate_name}_group_std"] = \
                beta_group_std
            df_tmp[f"beta_{target_parameter}_{covariate_name}_subject"] = \
                beta_subject[df_tmp[f"participant_id"].astype(int)]

            df_tmp[target_parameter] = df_tmp[target_parameter] + \
                                       (beta_subject[df_tmp[f"participant_id"].astype(int)] * df_tmp[
                                           f"{covariate_name}"])
            if cnt > 100:
                raise ValueError("Failed to sample parameters within bounds after 100 attempts")
            cnt += 1

            if np.all(df_tmp[target_parameter] < (param_bounds_buffered[1])) and \
                    np.all(df_tmp[target_parameter] > (param_bounds_buffered[0])):
                break

        # Update parameter_dicts
        group_params.update({
            f"beta_{target_parameter}_{covariate_name}_group_mean": \
                beta_group_mean,
            f"beta_{target_parameter}_{covariate_name}_group_std": \
                beta_group_std,
        })
        subject_params.update({
            f"beta_{target_parameter}_{covariate_name}_subject": beta_subject
        })

        # Update dataframe
        self.df = df_tmp
        self.group_params.update(group_params)
        self.subject_params.update(subject_params)
        return self.df, self.group_params, self.subject_params

    def inject_continuous_regression(self,
                                     target_parameter: str,
                                     covariate_name: str,
                                     group_beta_sign: str | None = "negative") -> tuple[pd.DataFrame, dict, dict]:
        """Injects continuous regression effects into a parameter of choice.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing trial-wise data
        model : str
            Name of the model being used
        target_parameter : str
            Parameter to inject regression effects into
        covariate_names : list[str]
            List of names for the continuous covariates to add
        beta_sign : str | None, optional
            Sign of the beta parameter, by default "negative"

        Returns
        -------
        df : pd.DataFrame
            DataFrame with added continuous regression effects
        group_params : dict
            Dictionary containing group-level parameters
        subject_params : dict
            Dictionary containing subject-level parameters
        """
        param_bounds_buffered, param_bounds_range, param_bounds_std_max = self.get_parameter_constraints(
            target_parameter)

        n_participants = len(self.df["participant_id"].unique())
        n_trials = len(self.df["trial"].unique())

        group_params = {}
        subject_params = {}

        cnt = 0
        while True:
            df_tmp = deepcopy(self.df)
            df_tmp[f"continuous_{covariate_name}"] = np.random.normal(-1, 1, size=(n_participants * n_trials))

            # Sample parameters
            if f"beta_{target_parameter}_{covariate_name}_group_mean" in self.group_params_supplied:
                beta_group_mean = self.group_params_supplied[f"beta_{target_parameter}_{covariate_name}_group_mean"]
            else:
                if group_beta_sign == "positive":
                    beta_group_mean = np.random.uniform(0.1,
                                                        self.continuous_regression_beta_mean_bounds[1])
                elif group_beta_sign == "negative":
                    beta_group_mean = np.random.uniform(self.continuous_regression_beta_mean_bounds[0],
                                                        -0.1)
                else:
                    beta_group_mean = np.random.uniform(self.continuous_regression_beta_mean_bounds[0],
                                                        self.continuous_regression_beta_mean_bounds[1])

            if f"beta_{target_parameter}_{covariate_name}_group_std" in self.group_params_supplied:
                beta_group_std = self.group_params_supplied[f"beta_{target_parameter}_{covariate_name}_group_std"]
            else:
                beta_group_std = np.random.uniform(0.05,
                                                   np.minimum(0.25, np.maximum(0.05,
                                                                               param_bounds_std_max)))

            if f"beta_{target_parameter}_{covariate_name}_subject" in self.subject_params_supplied:
                beta_subjects = self.subject_params_supplied[f"beta_{target_parameter}_{covariate_name}_subject"]
            else:
                beta_subjects = np.random.normal(loc=beta_group_mean,
                                                 scale=beta_group_std,
                                                 size=n_participants)

                # Update df
                df_tmp[f"beta_{target_parameter}_{covariate_name}_group_mean"] = beta_group_mean
                df_tmp[f"beta_{target_parameter}_{covariate_name}_group_std"] = beta_group_std
                df_tmp[f"beta_{target_parameter}_{covariate_name}_subject"] = beta_subjects[
                    df_tmp[f"participant_id"].astype(int).values]

            # Compute linear predictor (target_parameter)
            df_tmp[target_parameter] = df_tmp[target_parameter] + \
                                       (df_tmp[f"beta_{target_parameter}_{covariate_name}_subject"] * df_tmp[
                                           f"continuous_{covariate_name}"])

            if np.all(df_tmp[target_parameter] < (np.array(param_bounds_buffered[1]))) and \
                    np.all(df_tmp[target_parameter] > (np.array(param_bounds_buffered[0]))):
                break

            if cnt > 100:
                raise ValueError("Failed to sample parameters within bounds after 100 attempts")
            cnt += 1

        # Update parameter_dicts
        group_params.update({
            f"beta_{target_parameter}_{covariate_name}_group_mean": beta_group_mean,
            f"beta_{target_parameter}_{covariate_name}_group_std": beta_group_std,
        })
        subject_params.update({
            f"beta_{target_parameter}_{covariate_name}_subject": beta_subjects[
                df_tmp[f"participant_id"].astype(int).values]
        })

        self.df = df_tmp
        self.group_params.update(group_params)
        self.subject_params.update(subject_params)
        return self.df, self.group_params, self.subject_params

    def inject_sticky_choice(self,
                             target_parameter: str) -> tuple[pd.DataFrame, dict, dict]:
        """Injects trial-by-trial dependencies into a parameter based on previous trial responses.

        Parameters
        ----------
        target_parameter : str
            Parameter to inject dependencies into ('v' or 'z')

        Returns
        -------
        df_tmp : pd.DataFrame
            DataFrame with injected trial dependencies
        group_params : dict
            Dictionary of group-level parameters
        subject_params : dict
            Dictionary of subject-level parameters

        Raises
        ------
        ValueError
            If target_parameter is not 'v' or 'z'
            If parameters cannot be sampled within bounds after 100 attempts
        """
        if target_parameter not in ["v", "z"]:
            raise ValueError("Past dependence conceptually only implemented for drift 'v' or bias 'z': "
                             f" You specified {target_parameter}")

        param_bounds_buffered, param_bounds_range, param_bounds_std_max = \
            self.get_parameter_constraints(target_parameter)

        n_participants = len(self.df["participant_id"].unique())
        n_trials = len(self.df["trial"].unique())

        cnt = 0
        group_params = {}
        subject_params = {}
        while True:
            df_tmp = deepcopy(self.df)

            if f"beta_sticky_{target_parameter}_group_mean" in self.group_params_supplied:
                beta_sticky_group_mean = self.group_params_supplied[f"beta_sticky_{target_parameter}_group_mean"]
            else:
                beta_sticky_group_mean = np.random.uniform(0.05, param_bounds_std_max)

            if f"beta_sticky_{target_parameter}_group_std" in self.group_params_supplied:
                beta_sticky_group_std = self.group_params_supplied[f"beta_sticky_{target_parameter}_group_std"]
            else:
                beta_sticky_group_std = np.random.uniform(0.05,
                                                          np.maximum(0.05, param_bounds_std_max / 2))

            if f"beta_sticky_{target_parameter}_subject" in self.subject_params_supplied:
                beta_sticky_subjects = self.subject_params_supplied[f"beta_sticky_{target_parameter}_subject"]
            else:
                beta_sticky_subjects = np.random.normal(loc=beta_sticky_group_mean,
                                                        scale=beta_sticky_group_std,
                                                        size=n_participants)

            # Update parameter_dicts
            group_params.update({
                f"beta_sticky_{target_parameter}_group_mean": beta_sticky_group_mean,
                f"beta_sticky_{target_parameter}_group_std": beta_sticky_group_std,
            })
            subject_params.update({
                f"beta_sticky_{target_parameter}_subject": beta_sticky_subjects
            })

            # Update df
            df_tmp[f"beta_sticky_{target_parameter}_group_mean"] = \
                beta_sticky_group_mean
            df_tmp[f"beta_sticky_{target_parameter}_group_std"] = \
                beta_sticky_group_std
            df_tmp[f"beta_sticky_{target_parameter}_subject"] = \
                beta_sticky_subjects[df_tmp[f"participant_id"].astype(int).values]

            # Compute linear predictor (target_parameter)
            df_tmp["response_l1"] = 0
            df_tmp["response"] = 0
            for i in df_tmp["participant_id"].unique():
                displacement_tmp = 0
                for j in range(1, n_trials + 1, 1):

                    params_tmp = df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j),
                    self.model_params].to_dict(orient="records")[0]

                    if j > 1:
                        displacement_tmp = \
                        (df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j - 1), "response"].values * \
                         df_tmp.loc[(df_tmp["participant_id"] == i) & (
                                     df_tmp["trial"] == j), f"beta_sticky_{target_parameter}_subject"].values)[0]

                    # Apply displacement
                    params_tmp[f"{target_parameter}"] = params_tmp[f"{target_parameter}"] + displacement_tmp

                    df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j), f"{target_parameter}"] = \
                        params_tmp[f"{target_parameter}"]

                    sim_tmp = simulator(model=self.model,
                                        theta=params_tmp,
                                        n_samples=1)

                    df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j), "rt"] = \
                        sim_tmp['rts'].squeeze()
                    df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j), "response"] = \
                        sim_tmp['choices'].squeeze()

                    if j > 1:
                        df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j), "response_l1"] = \
                            df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j - 1), "response"].values

            if np.all(df_tmp[f"{target_parameter}"] < (param_bounds_buffered[1])) and \
                    np.all(df_tmp[f"{target_parameter}"] > (param_bounds_buffered[0])):
                break
            else:
                print(f"{target_parameter} is out of bounds, trying again, "
                      f"max {target_parameter} = {df_tmp[f'{target_parameter}'].max()} "
                      f"min {target_parameter} = {df_tmp[f'{target_parameter}'].min()}")
                print(f"allowable range = {param_bounds_buffered[0]} to {param_bounds_buffered[1]}")

            if cnt > 5:
                raise ValueError("Failed to sample parameters within bounds after 5 attempts")
            cnt += 1

        self.df = df_tmp
        self.group_params.update(group_params)
        self.subject_params.update(subject_params)
        self.state["sampled"] = True
        self.state["injected_sequential_dynamics"] = True
        return self.df, self.group_params, self.subject_params

    def inject_basic_rl_process(self):
        """Injects a basic reinforcement learning process into the dataset.

        This method injects a basic reinforcement learning process into the dataset,
        which is a simple two-choice process with a fixed learning rate.
        """

        n_participants = len(self.df["participant_id"].unique())
        n_trials = len(self.df["trial"].unique())

        cnt = 0
        group_params = {}
        subject_params = {}
        while True:
            df_tmp = deepcopy(self.df)
            # Compute linear predictor (target_parameter)
            df_tmp["response"] = 0.
            df_tmp["rt"] = 0.
            df_tmp["feedback"] = 0.
            df_tmp["correct"] = 0
            df_tmp["rl_q_val_m1"] = 0.
            df_tmp["rl_q_val_1"] = 0.

            rl_target_param = self.basic_rl_config["learnable_model_param"]
            for i in df_tmp["participant_id"].unique():
                q_val = np.array([0.5, 0.5])
                for j in range(1, n_trials + 1, 1):
                    # Get trial-parameters
                    params_tmp = df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j),
                    self.model_params].to_dict(orient="records")[0]
                    rl_params_tmp = df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j),
                    self.basic_rl_params].to_dict(orient="records")[0]

                    # Update drift based on rl-rule
                    params_tmp[rl_target_param] = rl_params_tmp["rl_scaler"] * (q_val[1] - q_val[0])

                    # Update drift in df
                    df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j), rl_target_param] = \
                        params_tmp[rl_target_param]

                    # Simulate action and rt
                    sim_tmp = simulator(model=self.model,
                                        theta=params_tmp,
                                        n_samples=1)

                    # Post-process action
                    ssm_action = sim_tmp["choices"].squeeze()
                    ssm_rt = sim_tmp["rts"].squeeze()

                    if ssm_action == -1:
                        action = 0
                        reward = np.random.normal(self.basic_rl_env['ch0_mean'],
                                                  self.basic_rl_env['ch0_std'])
                    elif ssm_action == 1:
                        action = 1
                        reward = np.random.normal(self.basic_rl_env['ch1_mean'],
                                                  self.basic_rl_env['ch1_std'])
                    else:
                        raise ValueError("ssm_action is not -1 or 1")

                    if self.basic_rl_env['ch0_mean'] > self.basic_rl_env['ch1_mean']:
                        action_correct = 0
                    elif self.basic_rl_env['ch0_mean'] < self.basic_rl_env['ch1_mean']:
                        action_correct = 1
                    else:
                        raise ValueError("ch0_mean and ch1_mean are equal")

                    if action == action_correct:
                        correct = 1
                    else:
                        correct = 0

                    # update q-values
                    q_val[action] += rl_params_tmp["rl_alpha"] * (reward - q_val[action])

                    # Update df
                    df_tmp.loc[(df_tmp["participant_id"] == i) & (df_tmp["trial"] == j), ["response", "rt", "feedback",
                                                                                          "correct", "rl_q_val_m1",
                                                                                          "rl_q_val_1"]] = \
                        [ssm_action, ssm_rt, reward, correct, q_val[0], q_val[1]]

            # Check that v is whithin bounds
            param_bounds_v_buffered, param_bounds_v_range, param_bounds_v_std_max = \
                self.get_parameter_constraints(rl_target_param)

            if np.all(df_tmp[rl_target_param] < (param_bounds_v_buffered[1])) and \
                    np.all(df_tmp[rl_target_param] > (param_bounds_v_buffered[0])):
                break
            else:
                print(f"{rl_target_param} is out of bounds, trying again, "
                      f"max {rl_target_param} = {df_tmp[rl_target_param].max()} "
                      f"min {rl_target_param} = {df_tmp[rl_target_param].min()}")
                print(f"allowable range = {param_bounds_v_buffered[0]} to {param_bounds_v_buffered[1]}")

            if cnt > 5:
                raise ValueError("Failed to sample parameters within bounds after 5 attempts")
            cnt += 1

        self.df = df_tmp
        self.group_params.update(group_params)
        self.subject_params.update(subject_params)
        self.state["sampled"] = True
        self.state["injected_sequential_dynamics"] = True
        return self.df, self.group_params, self.subject_params

    def make_simple_hierarchical_dataset(self,
                                         group_parameters_supplied: dict | None = None,
                                         subject_parameters_supplied: dict | None = None) -> tuple[
        pd.DataFrame, dict, dict]:
        """Generate a simple hierarchical dataset with parameters and simulated trials.

        This method creates a hierarchical dataset by first generating parameter sets,
        then injecting trials and attaching simulations.

        Parameters
        ----------
        group_parameters_supplied : dict | None, optional
            Group-level parameters to use for the simulation, by default None
        subject_parameters_supplied : dict | None, optional
            Subject-level parameters to use for the simulation, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame containing the hierarchical dataset with parameters and simulated data
        dict
            Dictionary containing group-level parameters
        dict
            Dictionary containing subject-level parameters
        """
        if group_parameters_supplied is not None:
            self.group_params_supplied.update(group_parameters_supplied)
        if subject_parameters_supplied is not None:
            self.subject_params_supplied.update(subject_parameters_supplied)

        _, _, _ = self.get_parameter_sets_hierarchical()

        _ = self.inject_trials()
        _ = self.attach_simulations_to_df()
        return self.df, self.group_params, self.subject_params

    def make_hierarchical_regression_dataset(self,
                                             categorical_target: str | None = "a",
                                             categorical_covariate: str | None = "cost_fail_condition",
                                             categorical_beta_sign: str | None = "positive",
                                             continuous_target: str | None = "v",
                                             continuous_covariate: str | None = "difficulty",
                                             continuous_beta_sign: str | None = "negative",
                                             group_parameters_supplied: dict | None = None,
                                             subject_parameters_supplied: dict | None = None) -> tuple[
        pd.DataFrame, dict, dict]:
        """Generate a hierarchical regression dataset with both categorical and continuous covariates.

        Parameters
        ----------
        categorical_target : str or None, optional
            Parameter to be modulated by categorical covariate, by default "a"
        categorical_covariate : str or None, optional
            Name of categorical covariate, by default "cost_fail_condition"
        categorical_beta_sign : str | None, optional
            Sign of the categorical beta parameter, by default "positive"
        continuous_target : str or None, optional
            Parameter to be modulated by continuous covariate, by default "v"
        continuous_covariate : str or None, optional
            Name of continuous covariate, by default "difficulty"
        continuous_beta_sign : str | None, optional
            Sign of the continuous beta parameter, by default "negative"
        group_parameters_supplied : dict | None, optional
            Group-level parameters to use for the simulation, by default None
        subject_parameters_supplied : dict | None, optional
            Subject-level parameters to use for the simulation, by default None

        Returns
        -------
        tuple[pd.DataFrame, dict, dict]
            DataFrame containing the hierarchical dataset with parameters and simulated data,
            Dictionary containing group-level parameters,
            Dictionary containing subject-level parameters

        Raises
        ------
        ValueError
            If categorical_target and continuous_target are the same parameter
        """
        if group_parameters_supplied is not None:
            self.group_params_supplied.update(group_parameters_supplied)
        if subject_parameters_supplied is not None:
            self.subject_params_supplied.update(subject_parameters_supplied)

        if (categorical_target is not None) and (continuous_target is not None):
            if categorical_target == continuous_target:
                raise ValueError("categorical_target and continuous_target cannot be the same")

        _, _, _ = self.get_parameter_sets_hierarchical()
        _ = self.inject_trials()

        if categorical_target is not None:
            _, _, _ = self.inject_categorical_regression(target_parameter=categorical_target,
                                                         covariate_name=categorical_covariate,
                                                         group_beta_sign=categorical_beta_sign)

        if continuous_target is not None:
            _, _, _ = self.inject_continuous_regression(target_parameter=continuous_target,
                                                        covariate_name=continuous_covariate,
                                                        group_beta_sign=continuous_beta_sign)

        _ = self.attach_simulations_to_df()
        return self.df, self.group_params, self.subject_params

    def make_hierarchical_regression_with_sticky_choice(self,
                                                        categorical_target: str | None = "a",
                                                        categorical_covariate: str | None = "cost_fail_condition",
                                                        categorical_beta_sign: str | None = "positive",
                                                        continuous_target: str | None = "v",
                                                        continuous_covariate: str | None = "difficulty",
                                                        continuous_beta_sign: str | None = "negative",
                                                        sticky_target: str = "z",
                                                        group_parameters_supplied: dict | None = None,
                                                        subject_parameters_supplied: dict | None = None) -> tuple[
        pd.DataFrame, dict, dict]:
        """Generate hierarchical regression data with sticky choice effects.

        Parameters
        ----------
        categorical_target : str or None, optional
            Parameter to apply categorical regression to. Default is "a"
        categorical_covariate : str or None, optional
            Name of categorical covariate. Default is "cost_fail_condition"
        continuous_target : str or None, optional
            Parameter to apply continuous regression to. Default is "v"
        continuous_covariate : str or None, optional
            Name of continuous covariate. Default is "difficulty"
        sticky_target : str, optional
            Parameter to apply sticky choice effects to. Default is "z"
        group_parameters_supplied : dict | None, optional
            Group-level parameters to use for the simulation, by default None
        subject_parameters_supplied : dict | None, optional
            Subject-level parameters to use for the simulation, by default None

        Returns
        -------
        tuple
            Contains:
            - pd.DataFrame: DataFrame with simulated data
            - dict: Group-level parameters
            - dict: Subject-level parameters

        Raises
        ------
        ValueError
            If categorical_target and continuous_target are the same parameter
        """
        if group_parameters_supplied is not None:
            self.group_params_supplied.update(group_parameters_supplied)
        if subject_parameters_supplied is not None:
            self.subject_params_supplied.update(subject_parameters_supplied)

        if (categorical_target is not None) and (continuous_target is not None):
            if categorical_target == continuous_target:
                raise ValueError("categorical_target and continuous_target cannot be the same")

        _, _, _ = self.get_parameter_sets_hierarchical()
        _ = self.inject_trials()

        if categorical_target is not None:
            _, _, _ = self.inject_categorical_regression(
                target_parameter=categorical_target,
                covariate_name=categorical_covariate,
                group_beta_sign=categorical_beta_sign)

        if continuous_target is not None:
            _, _, _ = self.inject_continuous_regression(
                target_parameter=continuous_target,
                covariate_name=continuous_covariate,
                group_beta_sign=continuous_beta_sign)

        # Note: This adds simulations to the df
        _, _, _ = self.inject_sticky_choice(target_parameter=sticky_target)

        return self.df, self.group_params, self.subject_params

    def make_rl_dataset(self,
                        group_parameters_supplied: dict | None = None,
                        subject_parameters_supplied: dict | None = None):
        """Generate a dataset for reinforcement learning models.

        This method creates a dataset with simulated data for reinforcement learning models.
        It includes simulated choices and rewards for multiple subjects and trials.

        Returns
        -------
        tuple
            Contains:
            - pd.DataFrame: DataFrame with simulated data
            - dict: Group-level parameters
            - dict: Subject-level parameters
        """

        if group_parameters_supplied is not None:
            self.group_params_supplied.update(group_parameters_supplied)
        if subject_parameters_supplied is not None:
            self.subject_params_supplied.update(subject_parameters_supplied)

        _, _, _ = self.get_parameter_sets_hierarchical(include_basic_rl=True)
        _ = self.inject_trials()
        _, _, _ = self.inject_basic_rl_process()

        return self.df, self.group_params, self.subject_params

    def save_data(self,
                  filename_base: str,
                  folder: str = "data") -> None:
        """Save simulation data and parameters to files.

        Parameters
        ----------
        filename_base : str
            Base filename to use for saving files. Will be appended with suffixes.
        folder : str, optional
            Folder to save files in, by default "data"
        columns_for_modeling : list[str] | None, optional
            List of column names to include in modeling dataset, by default None

        Raises
        ------
        ValueError
            If columns_for_modeling is None
        """
        # Make folder via pathlib
        Path(folder).mkdir(parents=True,
                           exist_ok=True)

        df_ordered, df_modeling = self.prep_dfs_for_saving()

        # Save Full df:
        df_ordered.to_parquet(f"{folder}/{filename_base}_full.parquet",
                              index=False)

        # Save df for modeling:
        df_modeling.to_parquet(f"{folder}/{filename_base}_modeling.parquet",
                               index=False)

        # Save the parameter dicts as one dict via pickle
        with open(f"{folder}/{filename_base}_parameters.pkl", "wb") as f:
            pickle.dump({"group_params": self.group_params,
                         "subject_params": self.subject_params},
                        f)

    @classmethod
    def load_data(cls,
                  filename_base: str,
                  folder: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Load saved simulation data and parameters from files.

        Parameters
        ----------
        filename_base : str
            Base filename used when saving files
        folder : str, optional
            Folder containing saved files, by default "data"

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, dict]
            Contains:
            - DataFrame with modeling data
            - DataFrame with full data
            - Dict containing group and subject parameters
        """
        with open(f"{folder}/{filename_base}_parameters.pkl", "rb") as f:
            parameters = pickle.load(f)

        df_modeling = pd.read_parquet(f"{folder}/{filename_base}_modeling.parquet")
        df_ordered = pd.read_parquet(f"{folder}/{filename_base}_full.parquet")
        return df_ordered, df_modeling, parameters
