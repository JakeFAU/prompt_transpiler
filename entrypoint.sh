#!/bin/sh
set -e

is_truthy() {
    case "$(printf "%s" "$1" | tr "[:upper:]" "[:lower:]")" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

otel_flag="${PRCOMP_USE_OPENTELEMETRY:-${USE_OPENTELEMETRY:-}}"

if [ -z "$otel_flag" ]; then
    otel_flag="true"
fi

if is_truthy "$otel_flag"; then
    exec opentelemetry-instrument \
        --traces_exporter "${OTEL_TRACES_EXPORTER:-otlp}" \
        --metrics_exporter "${OTEL_METRICS_EXPORTER:-console}" \
        --service_name "${OTEL_SERVICE_NAME:-prompt-compiler-api}" \
        --exporter_otlp_endpoint "${OTEL_EXPORTER_OTLP_ENDPOINT:-http://0.0.0.0:4317}" \
        prompt-compiler-api
fi

exec prompt-compiler-api
