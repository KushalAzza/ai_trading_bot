#!/bin/bash

ROOT="${PROJECT_ROOT:-$HOME/ai_trading_bot}"

export PATH="$ROOT/dependencies/miniconda3/bin:$PATH"
export CONDA_PREFIX="$ROOT/dependencies/miniconda3"

get_formatted_datetime() {
    date "+%Y-%m-%d %H:%M:%S"
}

cd "$ROOT" || exit 1

source "$ROOT/dependencies/miniconda3/bin/activate"
conda activate trading_env

declare -a holidays=(
    "2025-02-26" # Mahashivratri
    "2025-03-14" # Holi
    "2025-03-31" # Id-Ul-Fitr
    "2025-04-10" # Shri Mahavir Jayanti
    "2025-04-14" # Dr. Baba Saheb Ambedkar Jayanti
    "2025-04-18" # Good Friday
    "2025-05-01" # Maharashtra Day
    "2025-08-15" # Independence Day / Parsi New Year
    "2025-08-27" # Shri Ganesh Chaturthi
    "2025-10-02" # Mahatma Gandhi Jayanti/Dussehra
    "2025-10-21" # Diwali Laxmi Pujan
    "2025-10-22" # Balipratipada
    "2025-11-05" # Prakash Gurpurb Sri Guru Nanak Dev
    "2025-12-25" # Christmas
)

current_time=$(date +%H%M)
current_day=$(date +%u)
today=$(date +%Y-%m-%d)

is_holiday() {
    for holiday in "${holidays[@]}"; do
        if [ "$today" == "$holiday" ]; then
            return 0
        fi
    done
    return 1
}

if [ "$current_day" -le 5 ]; then
    if [ "$current_time" -ge 0920 ] && [ "$current_time" -le 1530 ]; then
        if is_holiday; then
            echo "$(get_formatted_datetime) - Trading holiday. Script not executed."
        else
            echo "$(get_formatted_datetime) - Starting script execution..."
            "$ROOT/dependencies/miniconda3/envs/trading_env/bin/python3" scripts/data_collection.py
            echo "$(get_formatted_datetime) - Script execution completed"
        fi
    else
        echo "$(get_formatted_datetime) - Outside market hours (9:20 AM - 3:30 PM). Script not executed."
    fi
else
    echo "$(get_formatted_datetime) - Weekend day. Script not executed."
fi
