import types

from pyvene.models.intervenable_base import (
    CollectIntervention,
    HandlerList,
    IntervenableModel,
    InterventionOutput,
    LambdaIntervention,
    do_intervention,
    get_batch_size,
)

from hypersteer.models.modules.interventions import PayloadInterventionOutput


def generate(
    self: IntervenableModel,
    base,
    sources: list | None = None,
    unit_locations: dict | None = None,
    source_representations: dict | None = None,
    intervene_on_prompt: bool = False,
    subspaces: list | None = None,
    output_original_output: bool | None = False,
    **kwargs,
):
    """
    Intervenable generation function that serves a
    wrapper to regular model generate calls.

    Currently, we support basic interventions **in the
    prompt only**. We will support generation interventions
    in the next release.

    TODO: Unroll sources and intervene in the generation step.

    Parameters:
    base:                The base example.
    sources:             A list of source examples.
    unit_locations:      The intervention locations of
                         base.
    activations_sources: A list of representations.
    intervene_on_prompt: Whether only intervene on prompt.
    **kwargs:            All other generation parameters.

    Return:
    base_output: the non-intervened output of the base
    input.
    counterfactual_outputs: the intervened output of the
    base input.
    """
    # TODO: forgive me now, i will change this later.
    activations_sources = source_representations
    if sources is not None and not isinstance(sources, list):
        sources = [sources]

    self._cleanup_states()

    self._intervene_on_prompt = intervene_on_prompt
    self._is_generation = True

    if not intervene_on_prompt and unit_locations is None:
        # that means, we intervene on every generated tokens!
        unit_locations = {"base": 0}

    # broadcast
    unit_locations = self._broadcast_unit_locations(
        get_batch_size(base), unit_locations
    )
    sources = [None] * len(self._intervention_group) if sources is None else sources
    sources = self._broadcast_sources(sources)
    activations_sources = self._broadcast_source_representations(activations_sources)
    subspaces = self._broadcast_subspaces(get_batch_size(base), subspaces)

    self._input_validation(
        base,
        sources,
        unit_locations,
        activations_sources,
        subspaces,
    )

    base_outputs = None
    if output_original_output:
        # returning un-intervened output
        base_outputs = self.model.generate(**base, **kwargs)

    set_handlers_to_remove = None
    try:
        # intervene
        if self.mode == "parallel":
            set_handlers_to_remove = self._wait_for_forward_with_parallel_intervention(
                sources,
                unit_locations,
                activations_sources,
                subspaces,
            )
        elif self.mode == "serial":
            set_handlers_to_remove = self._wait_for_forward_with_serial_intervention(
                sources,
                unit_locations,
                activations_sources,
                subspaces,
            )

        # run intervened generate
        counterfactual_outputs = self.model.generate(**base, **kwargs)

        collected_activations = []
        if self.return_collect_activations:
            for key in self.sorted_keys:
                if isinstance(self.interventions[key], CollectIntervention):
                    collected_activations += self.activations[key]
    except Exception as e:
        raise e
    finally:
        if set_handlers_to_remove is not None:
            set_handlers_to_remove.remove()
        self._is_generation = False
        self._cleanup_states(
            skip_activation_gc=(sources is None and activations_sources is not None)
            or self.return_collect_activations
        )

    if self.return_collect_activations:
        return (base_outputs, collected_activations), counterfactual_outputs

    return base_outputs, counterfactual_outputs


def _intervention_setter(
    self: IntervenableModel,
    keys,
    unit_locations_base,
    subspaces,
    intervention_additional_kwargs=None,  # passed by pyvene>=0.1.8; unused here
) -> HandlerList:
    """
    Create a list of setter handlers that will set activations
    """
    self._tidy_stateful_activations()

    handlers = []
    for key_i, key in enumerate(keys):
        intervention = self.interventions[key]
        module_hook = self.intervention_hooks[key]
        if unit_locations_base[0] is not None:
            self._batched_setter_activation_select[key] = [
                0 for _ in range(len(unit_locations_base[0]))
            ]  # batch_size

        def hook_callback(model, args, kwargs, output=None):
            # if it is None, we use it as adaptor.
            if unit_locations_base[key_i] is not None and self._is_generation:
                is_prompt = self._key_setter_call_counter[key] == 0
                if not self._intervene_on_prompt or is_prompt:
                    self._key_setter_call_counter[key] += 1
                if self._intervene_on_prompt ^ is_prompt:
                    return  # no-op
            if output is None:
                if len(args) == 0:  # kwargs based calls
                    # PR: https://github.com/frankaging/align-transformers/issues/11
                    # We cannot assume the dict only contain one element
                    output = kwargs[list(kwargs.keys())[0]]
                else:
                    output = args

            selected_output = self._gather_intervention_output(
                output, key, unit_locations_base[key_i]
            )
            # TODO: need to figure out why clone is needed
            if not self.is_model_stateless:
                selected_output = selected_output.clone()

            if isinstance(intervention, CollectIntervention):
                intervened_representation = do_intervention(
                    selected_output,
                    None,
                    intervention,
                    subspaces[key_i] if subspaces is not None else None,
                )
                # fail if this is not a fresh collect
                assert key not in self.activations

                self.activations[key] = intervened_representation
                # no-op to the output

            else:
                if not isinstance(self.interventions[key], LambdaIntervention):
                    if intervention.is_source_constant:
                        raw_intervened_representation = do_intervention(
                            selected_output,
                            None,
                            intervention,
                            subspaces[key_i] if subspaces is not None else None,
                        )
                        if isinstance(
                            raw_intervened_representation, InterventionOutput
                        ):
                            self.full_intervention_outputs.append(
                                raw_intervened_representation
                            )
                            intervened_representation = (
                                raw_intervened_representation.output
                            )
                        elif isinstance(
                            raw_intervened_representation, PayloadInterventionOutput
                        ):
                            # PATCH: pass thru generation mode hack
                            self.full_intervention_outputs.append(
                                raw_intervened_representation.payload
                            )
                            intervened_representation = (
                                raw_intervened_representation.output
                            )
                        else:
                            intervened_representation = raw_intervened_representation
                    else:
                        intervened_representation = do_intervention(
                            selected_output,
                            self._reconcile_stateful_cached_activations(
                                key,
                                selected_output,
                                unit_locations_base[key_i],
                            ),
                            intervention,
                            subspaces[key_i] if subspaces is not None else None,
                        )
                else:
                    # highly unlikely it's a primitive intervention type
                    intervened_representation = do_intervention(
                        selected_output,
                        self._reconcile_stateful_cached_activations(
                            key,
                            selected_output,
                            unit_locations_base[key_i],
                        ),
                        intervention,
                        subspaces[key_i] if subspaces is not None else None,
                    )
                if intervened_representation is None:
                    return

                # setter can produce hot activations for shared subspace interventions if linked
                if key in self._intervention_reverse_link:
                    self.hot_activations[self._intervention_reverse_link[key]] = (
                        intervened_representation.clone()
                    )

                if isinstance(output, tuple):
                    _ = self._scatter_intervention_output(
                        output[0],
                        intervened_representation,
                        key,
                        unit_locations_base[key_i],
                    )
                else:
                    _ = self._scatter_intervention_output(
                        output,
                        intervened_representation,
                        key,
                        unit_locations_base[key_i],
                    )

                self._intervention_state[key].inc_setter_version()

        handlers.append(module_hook(hook_callback, with_kwargs=True))

    return HandlerList(handlers)


def monkeypatch_ax_model_generate(model: IntervenableModel):
    model.generate = types.MethodType(generate, model)
    model._intervention_setter = types.MethodType(_intervention_setter, model)

    return model
