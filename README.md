This code was used to run a series of experiments for a keynote presentation in the SynthIR Workshop at SIGIR 2026. It is based on a challenge collection that I created for workshop participants. Unfortunately the passage collection file is too big to add to this repository, so I had to compress it and split it. There is a shell script called `rebuild_collection.sh` in the root directory that can be used to automatically reconstruct the original collection -- assuming you are using Linux or MacosX and have the xz compression program installed. If you are on windows, I would suggest that you install WSL and use a simple Ubuntu image to do it, or, if you understand anything I said above, you can do it using the Windows PowerShell if you really know what you are doing.

You will need to create a new anaconda environment using the following commands after you have reconstructed the test collection.
```bash
conda create -n "synthIR" python=3.10
conda activate synthIR
pip install -r requirements.txt
```

You should also run the program `build_dataset.py` in the src directory as there were some issues with the original collection. I cannot guarantee that this will work flawlessly since there are a set of hold out labels that I cannot release at the moment for the test set in the event that we run the challenge again in the future. (Sorry). It should be relatively easy to modify the build_dataset.py as needed. There is plenty of data available to generate your own build, val, and test dataset.

I have also added the pdf file of the slides used in the presentation in the root directory, called talk.pdf if you want more context.

My main goal here is to provide a reference to all of the experiments that I ran in the event that you have a clever way that can actually solve this surprisingly difficult problem.

J. Shane Culpepper

