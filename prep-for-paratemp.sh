#! /bin/bash

# exit on failure:
set -e

# For Python / conda
module load anaconda3
# For openbabel (file conversions)
module load wxwidgets/3.0.2
module load openbabel/2.4.1
# For GROMACS version
module load gcc/4.9.2
module load openmpi/1.6.4_gcc4.8.1
module load cuda/8.0
module load gromacs/2016.3
# Versioneer fails for old versions of git
module load git

# process command line args
USAGE=$(cat <<-END
$0 usage:
    (no arguments)  load all necessary modules and conda environment and then
                    exit

    -i          --install       use this argument to create a new 'paratemp' conda environment
                                and install needed packages and dependencies
    -s          --start         use this command to start jupyter lab
    -p port     --port port     use this port for Jupyter on SCC
                                If not given, it will ask for a port or a random one will be
                                generated. It must be in the range 1024-65535 (and likely not
                                a common number that someone else might pick).

    -d          --dryrun        use this command to tell conda to not actually install packages
                                (useful for testing only)
                                (also, NOT gauranteed to NOT install anything. Aka, be careful 
                                and don't just assume it'll be fine).
END
)

# Set default values
SETUP=FALSE
DRY=""
START=FALSE
PORT=NONE


while (( "$#" )); do
    # TODO add update option to update ParaTemp (and maybe everything else?)
    case $1 in
        -i | --install )
            # Setup conda environment and install
            SETUP=TRUE
            ;;
        -d | --dry-run | --dryrun )
            # Don't actually install with conda
            DRY="--dry-run"
            ;;
        -s | --start | --start-jupyter )
            # Start jupyter lab
            START=TRUE
            ;;
        -p | --port )
            # give port number
            shift
            PORT=$1
            if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
                echo port must be a number. Given: "$PORT" && exit 1;
            elif (( "$PORT" < 1024 )) || (( "$PORT" > 65535 )); then
                echo port must be between 1024 and 65535. Given: "$PORT" && exit 1;
            fi

        *)
            echo Unrecognized argument: $1
            echo "$USAGE"
            exit 1
            ;;
    esac; shift;
done

if [ $SETUP = TRUE ]; then
    # Make directory for installed files
    PT_DIR=".paratemp_install"
    if [ -d $PT_DIR ]; then
        cd $PT_DIR
    else
        mkdir $PT_DIR
        cd $PT_DIR
    fi
    # Get ParaTemp from GitHub
    if [ -d ParaTemp ]; then
        echo ParaTemp already downloaded
    else
        git clone https://github.com/theavey/ParaTemp.git
    fi
    # Create conda environment, install dependencies, activate the enviroment
    if [ -d ~/.conda/envs/paratemp ]; then
        echo paratemp conda environment already exists
    else
        conda create $DRY --yes -n paratemp -c conda-forge "python>=3.6" jupyterlab "blas=*=openblas" "pip>=18"
    fi
    source activate paratemp
    # Currently need latest gromacswrapper (until 0.8+)
    if [ -d GromacsWrapper ]; then
        echo GromacsWrapper already installed
    else
        git clone https://github.com/Becksteinlab/GromacsWrapper.git
        cd GromacsWrapper
        pip install .
        cd ..
    pip install gromacswrapper acpype
    conda install --yes -c conda-forge --file ParaTemp/requirements.txt
    conda install $DRY --yes -c omnia parmed
    # Install ParaTemp
    cd ParaTemp
    pip install -e .
    cd ..

    # Setup for Jupyter

    jupyter lab --generate-config
    if [[ "$PORT" == NONE ]]; then
        echo What port should be used on SCC for Jupyter? '(Hit Enter for random port)'
        read PORT
        if [ -z $PORT ]; then
            PORT=$(awk 'BEGIN{srand();print int(rand()*(63000-2000))+2000 }')
        fi
    fi
    echo Using port $PORT
    SSH_CONFIG_LINES=$(cat <<-END
	Host scc2 2 scc
	    User $USER
	    HostName scc2.bu.edu
	    Port 22
	    LocalForward 11111 localhost:$PORT
	    ForwardX11 yes
	END
	)
    SETUP_JUPYTER_CONFIG=$(cat <<-END
	import json
	from pathlib import Path
        path = Path('~/.jupyter/jupyter_config.json').expanduser()
	if path.is_file():
	    config = json.load(path.open('r'))
	else:
	    config = dict()
	notebookapp = config.get('NotebookApp', dict())
	notebookapp['port'] = $PORT
	notebookapp['open_browser'] = False
	config['NotebookApp'] = notebookapp
	if '$DRY':
	    print(f'jupyter_config.json would be {config}')
	else:
	    json.dump(config, path.open('w'), indent=2)
	END
	)
    python -c "$SETUP_JUPYTER_CONFIG"
    SETUP_GROMACSWRAPPER=$(cat <<-END
	import gromacs
	gromacs.config.setup()
	from pathlib import Path
	import errno
	path = Path('~/.gromacswrapper.cfg').expanduser()
	if not path.is_file():
	    raise OSError(errno.ENOENT, 'No .gromacswrapper.cfg found')
	lines = path.read_text().split('\\n')
	out_lines = list()
	for line in lines:
	    if 'append_suffix' in line:
	        line = 'append_suffix = no'
	    out_lines.append(line)
	if '$DRY':
	    print(f'.gromacswrapper.cfg would be {"\\n".join(out_lines)}')
	else:
	    path.write_text('\\n'.join(out_lines))
	END
	)
    python -c "$SETUP_GROMACSWRAPPER"

    # Setup port forwarding

    echo
    echo Almost done with all setup. Just four more steps to go before you 
    echo be up and running with Jupyter!
    echo First, on your local computer \(Not SCC\), you will need to edit a
    echo file. Either open up another terminal window \(or XQuartz, XTerm, 
    echo etc.\), or log out then make these changes.
    echo Open the file .ssh/config with your favorite editor \(or nano if you
    echo do not have one\). For example type \(followed by enter\)
    echo "      nano ~/.ssh/config"
    echo Then copy and paste the following text into that file:
    echo 
    echo "$SSH_CONFIG_LINES"
    echo
    echo Save the file and exit.
    echo Second, once that is done, you can log back in to scc just by typing
    echo "      ssh scc"
    echo followed by your password.
    echo Third, once you have logged back in to scc, run this script 
    echo again with '`-s`'
    echo "      $0 -s"
    echo Fourth and finally, in a web browser, go to
    echo "      https://localhost:11111"
    echo You will likely be required to copy and paste a long string of random 
    echo letters and numbers that will be shown in your window into the Jupyter
    echo login window.
    echo \(This can be avoided in the future by creating a password. Ask if 
    echo interested in this option.\)
    exit
else
    source activate paratemp
fi

if [ $START = TRUE ]; then
    # Thought about trying automated port forwarding. Don't think it's
    # possible.
    # read -r -a ip_array <<< $SSH_CLIENT
    # client_ip=${ip_array[0]}

    # Start Jupyter and wait for it to finish
    jupyter lab
fi

exit

