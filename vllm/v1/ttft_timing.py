# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec


class RequestTTFTTrace(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """Request-scoped TTFT timing data.

    The fields are additive across frontend / engine-core / worker stages.
    Every field is optional and expressed in nanoseconds.
    """

    api_preprocess_ns: int = 0
    ipc_in_decode_ns: int = 0
    engine_preprocess_ns: int = 0
    first_batch_load_kv_ns: int = 0


def copy_request_ttft_trace(
    trace: RequestTTFTTrace | None,
) -> RequestTTFTTrace | None:
    if trace is None:
        return None
    return RequestTTFTTrace(
        api_preprocess_ns=trace.api_preprocess_ns,
        ipc_in_decode_ns=trace.ipc_in_decode_ns,
        engine_preprocess_ns=trace.engine_preprocess_ns,
        first_batch_load_kv_ns=trace.first_batch_load_kv_ns,
    )


def merge_request_ttft_trace(
    trace: RequestTTFTTrace | None,
    update: RequestTTFTTrace | None,
) -> RequestTTFTTrace | None:
    if update is None:
        return trace
    if trace is None:
        return copy_request_ttft_trace(update)

    if update.api_preprocess_ns:
        trace.api_preprocess_ns = update.api_preprocess_ns
    if update.ipc_in_decode_ns:
        trace.ipc_in_decode_ns = update.ipc_in_decode_ns
    if update.engine_preprocess_ns:
        trace.engine_preprocess_ns = update.engine_preprocess_ns
    if update.first_batch_load_kv_ns:
        trace.first_batch_load_kv_ns = update.first_batch_load_kv_ns
    return trace
