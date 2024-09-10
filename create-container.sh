#!/bin/bash
usedPorts=$(netstat -latn --inet | awk '{print $4}' | awk -F : '{print $2}' | sort -u)
for port in {8988..8999}; do
  echo "trying port $port"
  echo "$usedPorts" | grep -q "^$port$" || break
done
echo "starting container with jupyter notebook on port $port"


USER_NAME=$(id -u -n)
jupyter_command="export PATH=${PATH}:/.local/bin; jupyter notebook --no-browser --ip=0.0.0.0 --notebook-dir /home/${USER_NAME} --NotebookApp.token=<YOUR PASSWORD> --port=${port}"

docker run -ti -d \
	--name "t-sne-feature-export-$USER_NAME" \
        -u "$(id -u)":"$(id -g)" \
	-v "/home/$USER_NAME:/home/$USER_NAME" \
        -v /home:/host-homes \
        -v /home:/tf/notebooks \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/shadow:/etc/shadow:ro \
	--workdir="/home/$USER_NAME/t-sne-features" \
        -p "$port":"$port" \
        nordar/stroke_perfusion:2.4.0-publish \
        /bin/bash $jupyter_command


