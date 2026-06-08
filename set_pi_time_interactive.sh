#!/usr/bin/env bash
set -euo pipefail

# Prompt for local Pacific time, set the Linux system clock, then write it
# to the selected RTC. Intended for field use when NTP and the external
# DS3231 are unavailable.

RTC_DEVICE="${RTC_DEVICE:-auto}"
TIMEZONE="${TIMEZONE:-America/Los_Angeles}"
LOG_TAG="${LOG_TAG:-manual-pi-time-set}"

log() {
    local message="$*"
    printf '%s %s\n' "$(date -Is 2>/dev/null || printf unknown-time)" "$message"
    logger -t "$LOG_TAG" "$message" 2>/dev/null || true
}

die() {
    log "ERROR: $*"
    exit 1
}

require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        die "Run as root: sudo ./set_pi_time_interactive.sh"
    fi
}

choose_rtc_device() {
    if [ "$RTC_DEVICE" != "auto" ] && [ -e "$RTC_DEVICE" ]; then
        return
    fi

    local fallback_device=""
    local rtc_path rtc_name rtc_device rtc_device_path

    for rtc_path in /sys/class/rtc/rtc*; do
        [ -e "$rtc_path" ] || continue

        rtc_name="$(cat "$rtc_path/name" 2>/dev/null || true)"
        rtc_device="/dev/$(basename "$rtc_path")"
        rtc_device_path="$(readlink -f "$rtc_path/device" 2>/dev/null || true)"
        log "Detected RTC candidate: $rtc_device (${rtc_name:-unknown}) ${rtc_device_path:-}"

        if [ -z "$fallback_device" ] && [ -e "$rtc_device" ]; then
            fallback_device="$rtc_device"
        fi

        case "$rtc_name" in
            *rpi*|*RPI*|*rp1*|*RP1*|*raspberry*|*Raspberry*)
                RTC_DEVICE="$rtc_device"
                log "Selected Raspberry Pi onboard RTC: $RTC_DEVICE (${rtc_name:-unknown})"
                return
                ;;
        esac

        case "$rtc_device_path" in
            *rpi_rtc*|*soc:rpi_rtc*|*rp1*)
                RTC_DEVICE="$rtc_device"
                log "Selected Raspberry Pi onboard RTC: $RTC_DEVICE (${rtc_name:-unknown})"
                return
                ;;
        esac
    done

    if [ -n "$fallback_device" ]; then
        RTC_DEVICE="$fallback_device"
        log "Could not identify the Pi onboard RTC by name; falling back to $RTC_DEVICE."
        return
    fi

    die "No RTC device found."
}

set_timezone_if_available() {
    if command -v timedatectl >/dev/null 2>&1; then
        timedatectl set-timezone "$TIMEZONE" || log "Could not set timezone to $TIMEZONE; continuing."
    fi
}

disable_ntp_temporarily() {
    if command -v timedatectl >/dev/null 2>&1; then
        timedatectl set-ntp false || true
    fi
}

enable_ntp_if_available() {
    if command -v timedatectl >/dev/null 2>&1; then
        timedatectl set-ntp true || true
    fi
}

date_arg_from_input() {
    local value="$1"
    local date_part time_part

    date_part="${value%% *}"
    time_part="${value#* }"
    printf '%s %s\n' "$date_part" "${time_part//-/:}"
}

validate_time_input() {
    local value="$1"
    [[ "$value" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}[[:space:]][0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]
}

prompt_for_time() {
    local entered=""

    while true; do
        printf 'Enter current local time in Pacific time (yyyy-mm-dd hh-mm-ss): ' >&2
        IFS= read -r entered

        if validate_time_input "$entered" && date -d "$(date_arg_from_input "$entered")" >/dev/null 2>&1; then
            printf '%s\n' "$entered"
            return
        fi

        printf 'Invalid time. Example: 2026-05-10 14-30-00\n' >&2
    done
}

main() {
    require_root
    choose_rtc_device
    set_timezone_if_available
    disable_ntp_temporarily

    local entered_time date_arg
    entered_time="$(prompt_for_time)"
    date_arg="$(date_arg_from_input "$entered_time")"

    log "Setting system time to local $TIMEZONE time: $entered_time"
    date -s "$date_arg"

    log "Writing system time to RTC $RTC_DEVICE."
    hwclock --rtc="$RTC_DEVICE" --systohc --utc

    log "System time is now: $(date -Is)"
    log "RTC $RTC_DEVICE is now: $(hwclock --rtc="$RTC_DEVICE" --show --utc)"
    enable_ntp_if_available
}

main "$@"
