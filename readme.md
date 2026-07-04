# CARLA Simulation Setup

## Install CARLA

Currently using [CARLA 0.9.14](https://carla.org/2022/12/23/release-0.9.14/).

As that version seems to run the most stable on my PC.

## Quick Start

1. **Launch CARLA Server**:
   Open a terminal and run:
   ```bash
   ./CarlaUE4.sh -ResX=1280 -ResY=720 -RenderOffScreen
   ```


 2. **Activate Conda Environment**:
    Open a separate terminal and activate the CARLA environment:
    ```bash
    conda activate carla
    ```

 3. **Run Python Script**
    Execute Python script:
    ```bash
    python3 <path-to-python-script>
    ```
