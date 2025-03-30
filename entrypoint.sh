set -e  # Exit on any error

echo "🚀 ENTRYPOINT_MODE set to: $ENTRYPOINT_MODE"

if [ "$ENTRYPOINT_MODE" = "api" ]; then
    echo "📡 Starting FastAPI server..."
    exec uvicorn src.api:app --host 0.0.0.0 --port 8000
elif [ "$ENTRYPOINT_MODE" = "cli" ]; then
    echo "🖥️ Running CLI mode..."
    exec python src/cli.py "$@"
elif [ "$ENTRYPOINT_MODE" = "streamlit" ]; then
    echo "🎨 Launching Streamlit app..."
    exec streamlit run src/app.py --server.fileWatcherType none
else
    echo "❌ ERROR: Invalid ENTRYPOINT_MODE. Use one of: api, cli, streamlit."
    exit 1
fi