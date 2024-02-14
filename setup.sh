
<<DESCRIPTION
This script automates the setup of a Python virtual environment.
It creates a virtual environment and then installs all required
dependencies from a requirements.txt file.
DESCRIPTION


GREEN='\033[32m'
YELLOW='\033[33m'
RESET='\033[0m'
RED='\033[31m'


spinner() {
    local pid=$1
    local message=$2
    local delay=0.1
    local spinstr='|/-\'

    printf "${YELLOW}$message${RESET}"
	tput civis
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf "${GREEN}%c${RESET}" "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b"
    done
    printf "\b"
	tput cnorm
}

check_error() {
    if [ $1 -ne 0 ]; then
        printf " ${RED}✗${RESET}\n"
        cat .install_cache
        printf "\n"
        rm .install_cache
        exit $1
    else
        printf " ${GREEN}✓${RESET}\n"
    fi
}

create_venv() {
    (test -d venv) || python -m venv venv 1>/dev/null 2>.install_cache &
    pid=$!
    spinner $pid "Setting up venv...  "
    wait $pid
    check_error $?
}

install_dependencies() {
    venv/bin/pip install -U pip 1>/dev/null 2>.install_cache && venv/bin/pip install -r requirements.txt 1>/dev/null 2>.install_cache &
    pid=$!
    spinner $pid "Installing dependencies...  "
    wait $pid
    check_error $?
    [ -f .install_cache ] && rm .install_cache
}

clean_venv() {
    if [ -d "venv" ]; then
        rm -rf venv &
        pid=$!
        spinner $pid "Removing venv... "
        wait $pid
        check_error $?
    else
        printf "${YELLOW}No virtual environment found to clean ${GREEN}✓${RESET}\n"
    fi
}

case "$1" in
    install)
        create_venv
        install_dependencies
        ;;
    clean)
        clean_venv
        ;;
    *)
        echo "Usage: $0 {install|clean}"
        exit 1
esac
