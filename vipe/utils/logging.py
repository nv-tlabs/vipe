# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import tqdm

disable_progress_bar: bool = False


def configure_logging() -> logging.Logger:
    """
    Configure the logging system. This will detach all loggers under vipe from the root.
    To use the package in a bigger project you probably don't want to call this function and instead manage logging yourself.
    """
    logger = logging.getLogger("vipe")

    # Define a custom logging handler to use tqdm.write
    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            tqdm.tqdm.write(msg)

    # Add the TqdmLoggingHandler to the logger
    handler = TqdmLoggingHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


def pbar(iterable, **kwargs):
    """
    A wrapper around tqdm.tqdm that disables the progress bar if the disable_progress_bar flag is set.
    """
    if disable_progress_bar:
        return iterable
    return tqdm.tqdm(iterable, **kwargs)
