
import numpy as np
import numpy.typing as npt
from typing import Any
from pandas import DataFrame


class ValidatableState:
    """Provides automated state validation for simulation components."""

    def _validate_params(self, expected_type: Any, error_msg : str):
        """
        A private helper method that takes a type e.g int, bool, etc
        and uses that type to check if the paramas (value) used by the 
        calling method match the expected type. 
        
        If the params passed does not match that expected type an error message is raised.

        Args:
            expected_type (Any): Takes a type e.g bool, etc and checks
                                if the params of the calling method match that type

            error_msg (str): A custom error message that can be added to better explain
                             the error. Note, it has a default custom message but this
                             adds more information to it
        """
        for attr, value in vars(self).items():
            if not isinstance(value, expected_type):
                        raise ValueError(
                            f"The parameter {attr} has a type of {type(value).__name__}",
                            f"{error_msg}"
                        )
    
    def validate_attributes_are_set(self):
        """Dynamically scans instance attributes to ensure none are left as None.
        
        Raises:
            ValueError: If any attribute (excluding internal private fields 
                other than '_seed') is None.
        """
        for attr, value in vars(self).items():

            # Skip internal state managers like _rng, but check configuration fields and _seed
            if attr.startswith('_') and attr != '_seed':
                continue
                
            if value is None:
                raise ValueError(
                    f"Configuration error: '{attr}' has not been set in {self.__class__.__name__}. "
                    f"Please run 'set_attributes()' before generating data."
                )


    def validate_attributes_are_numerical(self):
        """
        Validates if the values are numerical, if not raise a value error.

        Raises:
            ValueError: if any attribute/parameter is not an integer or a float
        """

        self._validate_params(expected_type=(int, float), 
                              error_msg="The value must be a float or int"
                              )
       
            
class Seed:
    def __init__(self, seed: int):
       self.seed = seed



class Bedroom(ValidatableState):
    """Configuration and simulation parameters for a housing bedroom model.

    This class manages house bedroom criteria, spatial sizing configurations, 
    and environmental noise parameters used during data processing or generation.

    Attributes:
        sq_ft_per_bedroom (float): Average square footage allocated per bedroom.
        bedroom_noise_mean (float): The mean decibel level or rating of bedroom noise.
        min_bedrooms (int): The minimum acceptable bedroom threshold per house.
        max_bedrooms (int): The maximum acceptable bedroom threshold per house.
        number_of_houses (int): The total count of houses simulated or evaluated.
        bedroom_noise_std (float): Standard deviation of the bedroom noise levels.
    """

    def __init__(self):
        self.sq_ft_per_bedroom   = None
        self.bedroom_noise_mean  = None
        self.min_bedrooms        = None
        self.max_bedrooms        = None
        self.number_of_houses    = None
        self.bedroom_noise_std   = None
        self.bedroom_noise_mean  = None
        self._seed               = None

    def bedrooms(self, area: float) -> np.ndarray:
        """Simulates the number of bedrooms for a collection of houses based on area.

        Calculates an initial bedroom count by dividing the total area by the 
        square footage per bedroom, adds Gaussian noise representing structural 
        variance, and clips the results to configured thresholds.

        Args:
            area: The total spatial square footage available to be divided.

        Returns:
            A NumPy array of integers representing the calculated bedroom counts 
            for each simulated house (length matches `number_of_houses`).

        Raises:
            ValueError: If any required configuration attributes are left as None.
        """
        self.validate_attributes_are_set()
        
        rng = np.random.default_rng(self._seed)

        return np.clip(
            np.round(
                area / self.sq_ft_per_bedroom 
                + rng.normal(
                    self.bedroom_noise_mean, 
                    self.bedroom_noise_std, 
                    self.number_of_houses
                )
            ),
            self.min_bedrooms,
            self.max_bedrooms
        )

    def set_attributes(
        self, 
        seed: int, 
        sq_ft_per_bedroom: int, 
        min_bedrooms: int, 
        max_bedrooms: int,
        number_of_houses: int,
        bedroom_noise_std: float = 0.7,
        bedroom_noise_mean: float = 0,
    ):
        """Initialises the simulation configurations and noise parameters.

        Args:
            seed: Integer seed to ensure reproducibility in random generation.
            sq_ft_per_bedroom: Average square footage allocated per bedroom.
            min_bedrooms: Lower bound constraint for bedroom counts.
            max_bedrooms: Upper bound constraint for bedroom counts.
            number_of_houses: Total count of houses to run simulations for.
            bedroom_noise_std: Standard deviation for the Gaussian noise. 
                Defaults to 0.7.
            bedroom_noise_mean: Mean offset value for the Gaussian noise. 
                Defaults to 0.
        """
        self.sq_ft_per_bedroom  = sq_ft_per_bedroom
        self.min_bedrooms       = min_bedrooms
        self.max_bedrooms       = max_bedrooms
        self.number_of_houses   = number_of_houses
        self.bedroom_noise_std  = bedroom_noise_std
        self.bedroom_noise_mean = bedroom_noise_mean
        self._seed              = seed


        
class Area(ValidatableState):
    """Configuration and simulation parameters for a housing area model.

    This class manages spatial boundary criteria, average sizing expectations, 
    and random distribution metrics used to generate house area data.

    Attributes:
        min_area (int): The lower bound floor area constraint.
        max_area (int): The upper bound floor area constraint.
        average_area (int): The target mean value for the house area distribution.
        areas_std (float): Standard deviation of the simulated house areas.
        number_of_houses (int): Total count of simulated houses to generate.
    """

    def __init__(self):
        self.min_area          = None
        self.max_area          = None
        self.average_area      = None
        self.areas_std         = None
        self.number_of_houses  = None
        self._seed             = None

    @property
    def area(self) -> np.ndarray:
        """Generates simulated house areas using a clipped normal distribution.

        Returns:
            A NumPy array of rounded integers representing the calculated floor 
            areas for each simulated house (length matches `number_of_houses`).

        Raises:
            ValueError: If any required configuration attributes are left as None.
        """
        self.validate_attributes_are_set()

        rng = np.random.default_rng(self._seed)
        return (
            rng.normal(self.average_area, self.areas_std, self.number_of_houses)
            .clip(self.min_area, self.max_area)
            .round()
        )

    def set_attributes(
        self,  
        seed: int,
        min_area: int, 
        max_area: int,
        average_area: int, 
        area_std: float, 
        number_of_houses: int = 500
    ):
        """Initialises the area simulation boundaries and distribution parameters.

        Args:
            seed: Integer seed to ensure reproducibility in random generation.
            min_area: Minimum floor area constraint.
            max_area: Maximum floor area constraint.
            average_area: Mean target floor area for the population.
            area_std: Standard deviation of structural sizing variance.
            number_of_houses: Total count of houses to run simulations for. 
                Defaults to 500.
        """
        self.min_area         = min_area
        self.max_area         = max_area
        self.average_area     = average_area
        self.areas_std        = area_std
        self.number_of_houses = number_of_houses
        self._seed            = seed

        self.validate_attributes_are_numerical()



class Age(ValidatableState):
    """Configuration and simulation parameters for a housing age model.

    This class manages temporal boundary constraints and handles generating 
    randomly distributed house ages.

    Attributes:
        min_age (int): The lower bound constraint for house age (in years).
        max_age (int): The upper bound constraint for house age (in years).
        num_of_houses (int): Total count of simulated houses to generate.
    """

    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.num_of_houses = None
        self._seed = None

    def set_attributes(self, min_age: int, max_age: int, num_of_houses: int, seed: int):
        """Initialises the house age simulation boundaries and sample constraints.

        Args:
            min_age: Minimum age constraint for the houses.
            max_age: Maximum age constraint for the houses.
            num_of_houses: Total count of houses to run simulations for.
            seed: Integer seed to ensure reproducibility in random generation.
        """
        self.min_age = min_age
        self.max_age  = max_age
        self.num_of_houses = num_of_houses
        self._seed = seed  
       
    @property
    def age(self) -> npt.NDArray[np.float64]:
        """Generates simulated house ages using a uniform distribution.

        Returns:
            A NumPy array of rounded floats representing the individual 
            calculated ages for each simulated house.

        Raises:
            ValueError: If any required configuration attributes are left as None.
        """
        self.validate_attributes_are_set()
        self.validate_attributes_are_numerical()

        rng = np.random.default_rng(self._seed)
        return rng.uniform(self.min_age, self.max_age, self.num_of_houses).round()




class Distance(ValidatableState):
    """Configuration and simulation parameters for a transit distance model.

    This class manages proximity constraints and handles generating randomly 
    distributed distances from houses to points of interest.

    Attributes:
        min_distance (float): The lower bound proximity constraint.
        max_distance (int): The upper bound proximity constraint.
        num_of_houses (int): Total count of simulated houses to generate.
    """

    def __init__(self):
        self.min_distance  = None
        self.max_distance  = None
        self.num_of_houses = None
        self._seed         = None

    @property
    def distance(self) -> np.ndarray:
        """Generates simulated house distances using a uniform distribution.

        Returns:
            A NumPy array of floats rounded to 1 decimal place representing 
            the calculated distances for each simulated house.

        Raises:
            ValueError: If any required configuration attributes are left as None.
        """
        self.validate_attributes_are_set()
        self.validate_attributes_are_numerical()

        rng = np.random.default_rng(self._seed)
        return rng.uniform(self.min_distance, self.max_distance, self.num_of_houses).round(1)

    def set_attributes(self, min_distance: float, max_distance: int, num_of_houses: int, seed: int):
        """Initialises the distance simulation boundaries and sample constraints.

        Args:
            min_distance: Minimum distance constraint.
            max_distance: Maximum distance constraint.
            num_of_houses: Total count of houses to run simulations for.
            seed: Integer seed to ensure reproducibility in random generation.
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_of_houses = num_of_houses
        self._seed  = seed




class Price(ValidatableState):
    """Configuration and calculation engine for a housing valuation model.

    This class manages baseline valuation rates, premium adjustments per asset 
    feature, depreciation metrics, and valuation noise variables.

    Attributes:
        base_price (float): The starting baseline price for an arbitrary house.
        price_per_sq_ft (float): Incremental price value added per square foot.
        price_per_bedroom (float): Incremental price value added per bedroom.
        price_reduction_per_year (float): Depreciation value lost per year of house age.
        price_reduction_per_mile (float): Valuation value lost per mile of transit distance.
        price_noise_mean (float): The mean offset value for valuation noise.
        price_noise_std (float): Standard deviation for calculation variance.
    """

    def __init__(self):
        self.base_price                = None
        self.price_per_sq_ft           = None
        self.price_per_bedroom         = None 
        self.price_reduction_per_year  = None 
        self.price_reduction_per_mile  = None
        self.price_noise_mean          = None
        self.price_noise_std           = None
        self._seed                     = None   

    def set_attributes(
        self, 
        base_price: float, 
        price_per_sq_ft: float,
        price_per_bedroom: float,
        price_reduction_per_year: float,
        price_reduction_per_mile: float,
        price_noise_mean: float,
        price_noise_std: float,
        seed: int,
    ):
        """Initialises valuation weights, depreciation rates, and noise settings.

        Args:
            base_price: The baseline starting valuation for a unit.
            price_per_sq_ft: Added dollar value metric per square foot.
            price_per_bedroom: Added dollar value premium per bedroom.
            price_reduction_per_year: Financial deduction penalty per year of age.
            price_reduction_per_mile: Financial deduction penalty per mile of distance.
            price_noise_mean: Mean target calculation offset for Gaussian noise.
            price_noise_std: Standard deviation of arbitrary evaluation noise.
            seed: Integer seed to ensure reproducibility in random generation.
        """
        self.base_price = base_price
        self.price_per_sq_ft = price_per_sq_ft
        self.price_per_bedroom = price_per_bedroom
        self.price_reduction_per_year  = price_reduction_per_year
        self.price_reduction_per_mile = price_reduction_per_mile
        self.price_noise_mean = price_noise_mean
        self.price_noise_std = price_noise_std
        self._seed = seed

    def price(
        self, 
        area: npt.NDArray[np.float64], 
        bedrooms: npt.NDArray[np.float64], 
        age: npt.NDArray[np.float64], 
        distance: npt.NDArray[np.float64], 
        num_of_houses: int
    ) -> npt.NDArray[np.float64]:
        """Calculates simulated housing prices using features and valuation weights.

        Applies positive premiums for space and bedrooms, subtracts deductions 
        for age and distance, and injects unique Gaussian variation parameters.

        Args:
            area: Collection of simulated house square footages.
            bedrooms: Collection of simulated bedroom counts.
            age: Collection of simulated structural house ages.
            distance: Collection of simulated transit distances.
            num_of_houses: Total count of housing evaluations to simulate.

        Returns:
            A NumPy array of floating-point values representing final house prices.

        Raises:
            ValueError: If any required valuation configurations are left as None.
        """
        self.validate_attributes_are_set()
        self.validate_attributes_are_numerical()

        rng = np.random.default_rng(self._seed)
     
        return (
            self.base_price 
            + (self.price_per_sq_ft * area)
            + (self.price_per_bedroom * bedrooms)
            - (self.price_reduction_per_year * age)  
            - (self.price_reduction_per_mile * distance)
            + rng.normal(
                self.price_noise_mean,
                self.price_noise_std,
                num_of_houses
            )
        )


    

class House(ValidatableState):
    """Orchestrator class coordinating a complete pipeline for housing generation.

    This class aggregates individual component sub-models (Area, Bedroom, Age, 
    Distance, and Price) to generate a synthetic dataset of cohesive housing properties.

    Attributes:
        num_of_houses (int): Total count of simulated houses to manage.
    """

    def __init__(self, num_of_houses: int, seed: int = 42):
        self._bedroom      = Bedroom()
        self._area         = Area()
        self._age          = Age()
        self._distance     = Distance()
        self._price        = Price()
        self.num_of_houses = num_of_houses
        self._seed_value   = seed
        self._rng          = np.random.default_rng(self._seed_value)

    def set_seed(self, seed: int):
        """Set the seed for the master random number generator.

        This ensures reproducibility across data generation processes.

        Args:
            seed : int
                The seed value used to initialize `np.random.default_rng`.
        """
        self._seed_value = seed 
        self._rng = np.random.default_rng(seed)

    @property
    def seed(self) -> int:
        """Returns the current master configuration seed value."""
        return self._seed_value
    
    def set_bedroom_attributes(
            self, 
            sq_ft_per_bedroom: int, 
            min_bedrooms: int, 
            max_bedrooms: int,
            bedroom_noise_std: float = 0.7,
            bedroom_noise_mean: float = 0.0, 
        ):
        """Configures structural bedroom sub-model constraints and noise layers.

        Args:
            sq_ft_per_bedroom: Average square footage allocated per bedroom.
            min_bedrooms: Lower bound constraint for bedroom counts.
            max_bedrooms: Upper bound constraint for bedroom counts.
            bedroom_noise_std: Standard deviation of structural bedroom variance. Defaults to 0.7.
            bedroom_noise_mean: Mean offset for structural bedroom variance noise. Defaults to 0.0.
        """
        self._bedroom.set_attributes(
             sq_ft_per_bedroom=sq_ft_per_bedroom, 
             bedroom_noise_mean=bedroom_noise_mean, 
             min_bedrooms=min_bedrooms, 
             max_bedrooms=max_bedrooms,
             number_of_houses=self.num_of_houses,
             bedroom_noise_std=bedroom_noise_std,
             seed=self._seed_value
        )

    def set_area_attributes(
        self, 
        min_area: int, 
        max_area: int,
        average_area: int, 
        area_std: float,                                   
    ):
        """Configures floor layout sizing metrics and footprint constraints.

        Args:
            min_area: Minimum floor area constraint.
            max_area: Maximum floor area constraint.
            average_area: Mean target floor area size.
            area_std: Standard deviation of property spatial variance.
        """
        self._area.set_attributes(
            min_area=min_area,
            max_area=max_area,
            average_area=average_area,
            area_std=area_std,
            number_of_houses=self.num_of_houses,
            seed=self._seed_value
        )

    def set_age_attributes(self, min_age: int, max_age: int):
        """Configures property duration boundary targets.

        Args:
            min_age: Minimum age constraint for structural properties.
            max_age: Maximum age constraint for structural properties.
        """
        self._age.set_attributes(
            min_age=min_age, 
            max_age=max_age, 
            num_of_houses=self.num_of_houses,
            seed=self._seed_value
        )

    def set_distance_attributes(self, min_distance: float, max_distance: int):
        """Configures geographical community proximity guidelines.

        Args:
            min_distance: Minimum travel distance bounds.
            max_distance: Maximum travel distance bounds.
        """
        self._distance.set_attributes(
            min_distance=min_distance,
            max_distance=max_distance,
            num_of_houses=self.num_of_houses,
            seed=self._seed_value
        )

    def set_price_attributes(
        self,
        base_price: float, 
        price_per_sq_ft: float,
        price_per_bedroom: float,
        price_reduction_per_year: float,
        price_reduction_per_mile: float,
        price_noise_mean: float,
        price_noise_std: float,
    ):
        """Configures assessment multipliers, depreciation penalties, and variation targets.

        Args:
            base_price: Baseline starting evaluation point.
            price_per_sq_ft: Rate metric premium added per square foot.
            price_per_bedroom: Rate metric premium added per bedroom.
            price_reduction_per_year: Financial deduction penalty applied per house year.
            price_reduction_per_mile: Financial deduction penalty applied per travel mile.
            price_noise_mean: Mean evaluation variation offset.
            price_noise_std: Standard deviation of calculation valuation variations.
        """
        self._price.set_attributes(
            base_price=base_price,
            price_per_sq_ft=price_per_sq_ft,
            price_per_bedroom=price_per_bedroom,
            price_reduction_per_year=price_reduction_per_year,
            price_reduction_per_mile=price_reduction_per_mile,
            price_noise_mean=price_noise_mean,
            price_noise_std=price_noise_std,
            seed=self._seed_value
        )

    @property
    def area(self) -> np.ndarray:
        """Retrieves generated data reflecting calculated room surface footprints."""
        return self._area.area

    @property
    def bedrooms(self) -> np.ndarray:
        """Retrieves generated data reflecting calculated structural bedroom rooms."""
        return self._bedroom.bedrooms(area=self.area)

    @property
    def age(self) -> np.ndarray:
        """Retrieves generated data reflecting structural building years."""
        return self._age.age  

    @property
    def distance(self) -> np.ndarray:
        """Retrieves generated data reflecting structural transit miles."""
        return self._distance.distance

    @property
    def price(self) -> npt.NDArray[np.float64]:
        """Calculates final combined evaluation metrics mapping across generated variables."""
        return self._price.price(
            area=self.area, 
            bedrooms=self.bedrooms,
            age=self.age,
            distance=self.distance,
            num_of_houses=self.num_of_houses
        )

    @property
    def min_age(self) -> int:
        """Exposes configured sub-component minimal age constraint."""
        return self._age.min_age

    @property
    def max_age(self) -> int:
        """Exposes configured sub-component maximal age constraint."""
        return self._age.max_age

    @property
    def min_distance_miles(self) -> float:
        """Exposes configured sub-component minimal proximity constraint."""
        return self._distance.min_distance 

    @property
    def max_distance_miles(self) -> int:
        """Exposes configured sub-component maximal proximity constraint."""
        return self._distance.max_distance

    @property
    def min_bedrooms(self) -> int:
        """Exposes configured sub-component minimal bedroom constraint."""
        return self._bedroom.min_bedrooms

    @property
    def max_bedrooms(self) -> int:
        """Exposes configured sub-component maximal bedroom constraint."""
        return self._bedroom.max_bedrooms

    def to_dataframe(self, 
                        add_random_noise = False, 
                        feature_noise_mean = 1, 
                        feature_noise_std = 1,
                         round_random_noise: int = 2) -> DataFrame:
        
        """Compiles all generated features into a structured Pandas DataFrame.

        Retrieves vector calculations simultaneously across every metric system 
        and bundles them with labeled data attributes.

        The caller of the method can also choose to add random noise to data.

        Args:
            add_random_noise (bool) : A boolean flag that adds random noise to the data
            feature_noise_mean (int): The mean value for the random noise
            feature_noise_std (int): The std value for the random noise
            round_random_noise (int): The number of decimals to round the nose to

        Returns:
            A Pandas DataFrame containing area, bedrooms, age, distance, 
            and final simulated prices for all items a random noise is 
            also added if the `add_random_noise` is set to True.
        """
        if not isinstance(add_random_noise, bool):
            raise TypeError(f"The attr 'add_random_noise' must be a bool. Got type {type(add_random_noise).__name__} ")

        if not isinstance(round_random_noise, int):
             raise TypeError(f"The attr 'round_random_noise' must be a int. Got type {type(round_random_noise).__name__} ")

        if not isinstance(feature_noise_mean, (int, float)) or not isinstance(feature_noise_std, int):
            raise TypeError(
                f"The feature_noise_mean or feature_noise_std is not a number or a float ",
                f"feature_noise_mean is {type(feature_noise_mean).__name__} "
                f"feature_nose_std is {type(feature_noise_std).__name__}"
            )

     
        if add_random_noise:
            random_noise = self._rng.normal(feature_noise_mean, 
                                            feature_noise_std, 
                                            self.num_of_houses).round(round_random_noise)
            return  DataFrame({
                "area": self.area,
                "bedrooms": self.bedrooms,
                "age": self.age,
                "distance": self.distance,
                "random_noise": random_noise,
                "price": self.price
            })


        # return without the random noise
        return DataFrame({
                "area": self.area,
                "bedrooms": self.bedrooms,
                "age": self.age,
                "distance": self.distance,
                "price": self.price
            })
