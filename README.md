[![CI](https://github.com/DiamondLightSource/xrpd-toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamondLightSource/xrpd-toolbox/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/DiamondLightSource/xrpd-toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamondLightSource/xrpd-toolbox)
[![PyPI](https://img.shields.io/pypi/v/xrpd-toolbox.svg)](https://pypi.org/project/xrpd-toolbox)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# xrpd_toolbox

A repository for useful python data analysis tools for the X-ray powder diffraction beamlines at Diamond Light Source

This is where you should write a short paragraph that describes what your module does,
how it does it, and why people should use it.

What            | Where
:---:           | :---:
Source          | <https://github.com/DiamondLightSource/xrpd-toolbox>
PyPI            | `pip install xrpd-toolbox`
Docker          | `docker run ghcr.io/diamondlightsource/xrpd-toolbox:latest`
Releases        | <https://github.com/DiamondLightSource/xrpd-toolbox/releases>


```python
from xrpd_toolbox import __version__

print(f"Hello xrpd_toolbox {__version__}")
```

Here are some useful things for

```python
from xrpd_toolbox.utils.messenger import Messenger

client = Messenger("i15-1", broker="rabbitmq", username="guest", password="guest") #this will connect to i15-1's raabitmq

client.send_message("/topic/public.worker.event", "MY MESSAGE")

client.listen() #to listen to what is happening on the default destinations

DEFAULT_DESTINATIONS = [
    "/topic/public.worker.event",
    "/topic/gda.messages.scan",
]

my_destinations = ["/topic/my_dest"]

#if you want to listen on other you can use:

client = Messenger("i15-1", broker="rabbitmq", username="guest", password="guest", destinations=my_destinations)

#if you want to connect to a specific message bus too:

client = Messenger(host=MY_HOST, port=MY_PORT, username="guest", password="guest", destinations=my_destinations)

```
