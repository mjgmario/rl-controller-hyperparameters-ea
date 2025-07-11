from inspect import signature
from typing import Any, Dict, List, Optional, Type, Union

from environment.observable import (
    AveragePopFitness,
    AvgDistanceFromBest,
    BestFitnessNorm,
    Entropy,
    FitnessImprovement,
    GenerationLog,
    GenotypicDiversity,
    Observable,
    Stagnation,
    StdFitness,
)


class ObservableFactory:
    """
    Factory to build Observable instances from names or existing objects,
    applying constructor parameters and attribute overrides.
    """

    _registry: Dict[str, Type[Observable]] = {
        "generation_log": GenerationLog,
        "average_pop_fitness": AveragePopFitness,
        "entropy": Entropy,
        "std_fitness": StdFitness,
        "fitness_improvement": FitnessImprovement,
        "stagnation": Stagnation,
        "genotypic_diversity": GenotypicDiversity,
        "best_fitness_norm": BestFitnessNorm,
        "avg_distance_from_best": AvgDistanceFromBest,
    }

    @classmethod
    def build(
        cls,
        entries: List[Union[str, Observable]],
        config: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Observable]:
        """
        Construct a list of Observable instances.

        entries: names of observables (strings) or existing Observable instances.
        config: mapping from observable name to constructor keyword args and attribute overrides.
        """
        config = config or {}
        instances: List[Observable] = []

        for entry in entries:
            if isinstance(entry, str):
                name = entry
                cls_type = cls._registry.get(name)
                if cls_type is None:
                    raise ValueError(f"Unknown observable '{name}'")

                full_config = config.get(name, {})

                # Separate constructor args and attributes
                constructor_args = set(signature(cls_type.__init__).parameters.keys())
                constructor_args.discard("self")
                init_kwargs = {
                    k: v for k, v in full_config.items() if k in constructor_args
                }
                attr_overrides = {
                    k: v for k, v in full_config.items() if k not in constructor_args
                }

                # Instantiate observable
                obs = cls_type(**init_kwargs)

                # Apply overrides
                for attr, val in attr_overrides.items():
                    if hasattr(obs, attr):
                        setattr(obs, attr, val)

                instances.append(obs)

            elif isinstance(entry, Observable):
                instances.append(entry)
            else:
                raise ValueError(
                    f"Invalid observable entry: {entry!r}; must be name or Observable"
                )

        return instances
