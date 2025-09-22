# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import os

is_enabled = False
if os.getenv("USE_PERFLIB", "1") == "1":
    print("Enabled the use of perflib.")
    is_enabled = True
