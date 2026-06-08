#!/usr/bin/env bash
set -euo pipefail

# Sync policy:
#   1. Try to get network/NTP time.
#   2. If NTP sync succeeds, write system time to the selected RTC.
#   3. In all cases, set system time from the selected RTC.
#
# For the current field setup, "auto" prefers the Raspberry Pi 5 onboard RTC.
# You can override with RTC_DEVICE=/dev/rtcN if needed.

NTP_WAIT_SECONDS="${NTP_WAIT_SECONDS:-60}"
NTP_POLL_SECONDS="${NTP_POLL_SECONDS:-3}"
RTC_DEVICE="${RTC_DEVICE:-auto}"
TIMEZONE="${TIMEZONE:-America/Los_Angeles}"
LOG_TAG="${LOG_TAG:-pi-rtc-time-sync}"

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
        die "Run as root. hwclock and system time changes require root privileges."
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
    if [ -z "$TIMEZONE" ]; then
        log "TIMEZONE is empty; skipping timezone setup."
        return
    fi

    if ! command -v timedatectl >/dev/null 2>&1; then
        log "timedatectl is not available; skipping timezone setup."
        return
    fi

    local current_timezone
    current_timezone="$(timedatectl show -p Timezone --value 2>/dev/null || true)"
    if [ "$current_timezone" = "$TIMEZONE" ]; then
        log "Timezone is already set to $TIMEZONE."
        return
    fi

    log "Setting timezone to $TIMEZONE."
    timedatectl set-timezone "$TIMEZONE" || log "Could not set timezone to $TIMEZONE; continuing."
}

hwclock_rtc_arg() {
    printf -- '--rtc=%s' "$RTC_DEVICE"
}

enable_ntp_if_available() {
    if command -v timedatectl >/dev/null 2>&1; then
        log "Requesting NTP via timedatectl."
        timedatectl set-ntp true || log "timedatectl set-ntp true failed; continuing."
    fi
}

timedatectl_synced() {
    command -v timedatectl >/dev/null 2>&1 || return 1

    local value
    value="$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)"
    case "$value" in
        yes|true|1) return 0 ;;
    esac

    value="$(timedatectl show -p SystemClockSynchronized --value 2>/dev/null || true)"
    case "$value" in
        yes|true|1) return 0 ;;
    esac

    return 1
}

chrony_synced() {
    command -v chronyc >/dev/null 2>&1 || return 1
    chronyc -n tracking 2>/dev/null | grep -q "Leap status[[:space:]]*: Normal"
}

ntp_is_synced() {
    timedatectl_synced || chrony_synced
}

wait_for_ntp_sync() {
    local waited=0

    while [ "$waited" -le "$NTP_WAIT_SECONDS" ]; do
        if ntp_is_synced; then
            return 0
        fi

        sleep "$NTP_POLL_SECONDS"
        waited=$((waited + NTP_POLL_SECONDS))
    done

    return 1
}

rtc_show() {
    hwclock "$(hwclock_rtc_arg)" --show || true
}

system_to_rtc() {
    log "Writing NTP-synchronized system time to RTC $RTC_DEVICE."
    hwclock "$(hwclock_rtc_arg)" --systohc --utc
}

rtc_to_system() {
    log "Setting system time from RTC $RTC_DEVICE."
    hwclock "$(hwclock_rtc_arg)" --hctosys --utc
}

main() {
    require_root
    choose_rtc_device
    set_timezone_if_available

    log "Using RTC device: $RTC_DEVICE"
    log "RTC before sync: $(rtc_show)"

    enable_ntp_if_available

    if wait_for_ntp_sync; then
        log "NTP sync detected."
        system_to_rtc
    else
        log "NTP sync was not available after ${NTP_WAIT_SECONDS}s."
    fi

    rtc_to_system
    log "RTC after sync: $(rtc_show)"
    log "System time after sync: $(date -Is)"
}

main "$@"
