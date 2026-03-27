import { Toaster } from "react-hot-toast";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Header }        from "./components/layout/Header";
import { WorldMap }      from "./components/map/WorldMap";
import { ForecastPanel } from "./components/forecast/ForecastPanel";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 2 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="h-screen w-screen flex flex-col bg-bg-primary
                      text-text-primary overflow-hidden">
        <Header />

        {/* Main content — map left, forecast panel right */}
        <div className="flex-1 flex overflow-hidden">
          {/* Map — fills remaining space */}
          <div className="flex-1 relative">
            <WorldMap />
          </div>

          {/* Forecast panel — fixed width right sidebar */}
          <div
            className="w-[420px] flex-shrink-0 border-l border-border
                       bg-bg-primary overflow-hidden flex flex-col
                       max-lg:w-80 max-md:hidden"
          >
            <ForecastPanel />
          </div>
        </div>
      </div>

      <Toaster
        position="bottom-center"
        toastOptions={{
          style: {
            background: "#161b22",
            color:      "#e6edf3",
            border:     "1px solid #30363d",
            fontSize:   "13px",
          },
        }}
      />
    </QueryClientProvider>
  );
}
