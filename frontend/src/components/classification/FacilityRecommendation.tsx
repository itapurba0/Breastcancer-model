import { useState, useCallback } from "react";
import {
  MapPin, Phone, ExternalLink, Navigation, Hospital,
  Loader2, Search, AlertCircle
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { classifierApi } from "@/lib/api";

interface Facility {
  id: string;
  name: string;
  type: string;
  specialties?: string[];
  address: string;
  city?: string;
  state?: string;
  phone?: string;
  website?: string;
  tier?: string;
  distance_km?: number | null;
  relevance_reason?: string;
  rating?: number;
  total_ratings?: number;
  open_now?: boolean;
  location?: { lat: number; lng: number };
}

interface FacilityRecommendationProps {
  prediction: string;
  confidence: number;
  inconclusive?: boolean;
}

const facilityTypeConfig: Record<string, { label: string; bg: string; text: string; border: string }> = {
  cancer_center: { label: "Cancer Center", bg: "bg-red-50", text: "text-red-700", border: "border-red-200" },
  diagnostic_center: { label: "Diagnostic Center", bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200" },
  general_hospital: { label: "General Hospital", bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
  hospital: { label: "Hospital", bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200" },
};

const CANCER_RE = /cancer|oncology|tumou?r|chemotherapy|radiation|surgical|onco/i;
const MAMMOGRAM_RE = /mammogra|breast|scan|imaging|radiology|sonography|ultrasound|diagnostic|screening/i;

function isRelevantFacility(f: Facility): boolean {
  if (f.type === "cancer_center") return true;
  if (f.type === "hospital" && CANCER_RE.test(f.name)) return true;
  if (f.type === "diagnostic_center" && MAMMOGRAM_RE.test(f.name)) return true;
  return false;
}

const linkClasses = "inline-flex items-center gap-1.5 text-xs font-semibold text-primary hover:text-primary/80 transition-colors duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 rounded-sm";

const FacilityRecommendation = ({ prediction, confidence, inconclusive }: FacilityRecommendationProps) => {
  const [city, setCity] = useState("");
  const [facilities, setFacilities] = useState<Facility[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [source, setSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [isDetectingLocation, setIsDetectingLocation] = useState(false);

  const searchQuery = useCallback(() => {
    if (inconclusive) return "diagnostic centre mammography breast imaging";
    if (prediction === "malignant") return "cancer hospital oncology centre";
    return "diagnostic centre mammography breast screening";
  }, [prediction, inconclusive]);

  const fetchCuratedFallback = useCallback(async (lat?: number, lng?: number, searchCity?: string) => {
    try {
      const res = await classifierApi("/facilities/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction,
          confidence,
          inconclusive,
          city: searchCity || undefined,
          lat,
          lng,
          limit: 15,
        }),
      });
      if (!res.ok) return false;
      const data = await res.json();
      const results = data.recommendations || [];
      if (results.length === 0) {
        setError("No facilities found in this area.");
        return false;
      }
      setFacilities(results);
      setSource("curated");
      setShowResults(true);
      return true;
    } catch {
      setError("Unable to fetch facility information. Please try again.");
      return false;
    }
  }, [prediction, confidence, inconclusive]);

  const fetchFromGoogle = useCallback(async (lat?: number, lng?: number, searchCity?: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const baseQuery = searchQuery();
      const query = searchCity ? `${baseQuery} in ${searchCity}` : baseQuery;

      const res = await classifierApi("/facilities/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, lat, lng, radius: 20000 }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      if (data.source === "unavailable") throw new Error("Google Places API not configured");

      let results = data.recommendations || [];
      results = results.filter(isRelevantFacility);

      if (results.length === 0) {
        return await fetchCuratedFallback(lat, lng, searchCity);
      }

      setFacilities(results);
      setSource("google");
      setShowResults(true);
    } catch {
      await fetchCuratedFallback(lat, lng, searchCity);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery, fetchCuratedFallback]);

  const handleCitySearch = () => {
    if (city.trim()) {
      fetchFromGoogle(undefined, undefined, city.trim());
    }
  };

  const handleGeolocation = () => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser.");
      return;
    }

    setIsDetectingLocation(true);
    setError(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setIsDetectingLocation(false);
        fetchFromGoogle(position.coords.latitude, position.coords.longitude, city.trim() || undefined);
      },
      () => {
        setIsDetectingLocation(false);
        if (city.trim()) {
          fetchFromGoogle(undefined, undefined, city.trim());
        } else {
          setError("Location access denied. Please enter your city manually.");
        }
      },
      { enableHighAccuracy: false, timeout: 10000 }
    );
  };

  const typeConfig = (type: string) => facilityTypeConfig[type] || facilityTypeConfig.hospital;

  return (
    <div className="space-y-4">
      {/* Location Input Panel */}
      {!showResults && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl p-5 bg-primary/5 border border-primary/10 space-y-4"
        >
          <div className="flex items-center gap-2">
            <Hospital className="h-4 w-4 text-primary" />
            <h4 className="text-sm font-bold text-foreground font-sans">Find nearby facilities</h4>
          </div>
          <p className="text-xs text-muted-foreground font-sans">
            Based on your scan results, we can recommend specialized medical facilities near you.
          </p>

          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Enter your city"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCitySearch()}
                className="pl-9 h-10 rounded-full text-sm"
              />
            </div>
            <div className="flex gap-2">
              <Button
                onClick={handleCitySearch}
                disabled={!city.trim() || isLoading}
                size="sm"
                className="h-10 px-4 rounded-full bg-primary text-white hover:bg-primary/90 text-sm font-semibold"
              >
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                <span className="ml-1.5 hidden sm:inline">Search</span>
              </Button>
              <Button
                onClick={handleGeolocation}
                disabled={isLoading || isDetectingLocation}
                size="sm"
                variant="outline"
                className="h-10 px-4 rounded-full text-sm font-semibold"
              >
                {isDetectingLocation ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Navigation className="h-4 w-4" />
                )}
                <span className="ml-1.5 hidden sm:inline">Near me</span>
              </Button>
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-2 p-3 rounded-xl bg-red-50 border border-red-200" role="alert" aria-live="assertive">
              <AlertCircle className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
              <p className="text-xs text-red-700 font-sans">{error}</p>
            </div>
          )}
        </motion.div>
      )}

      {/* Results */}
      <AnimatePresence>
        {showResults && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-3"
          >
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Hospital className="h-4 w-4 text-primary" />
                <h4 className="text-sm font-bold text-foreground font-sans">
                  {source === "curated" ? "Recommended facilities" : "Nearby facilities"}
                </h4>
                {facilities.length > 0 && (
                  <span className="text-xs text-muted-foreground font-mono bg-muted px-2 py-0.5 rounded-full tabular-nums">
                    {facilities.length}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {source === "curated" && (
                  <span className="text-xs text-muted-foreground font-sans bg-muted px-2 py-0.5 rounded-full">
                    Database
                  </span>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setShowResults(false);
                    setFacilities([]);
                    setError(null);
                  }}
                  className="h-8 px-3 text-xs rounded-full cursor-pointer"
                >
                  Change location
                </Button>
              </div>
            </div>

            {/* Facility Cards */}
            {facilities.length === 0 ? (
              <div className="text-center py-6 rounded-2xl bg-muted/50 border border-brand/10">
                <Hospital className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-sm text-muted-foreground font-sans">No facilities found for this area.</p>
                <p className="text-xs text-muted-foreground font-sans mt-1">Try a different city or use "Near me".</p>
              </div>
            ) : (
              <div className="max-h-[400px] overflow-y-auto scrollbar-hidden space-y-2">
                {facilities.map((facility, i) => {
                  const config = typeConfig(facility.type);
                  return (
                    <motion.div
                      key={facility.id || i}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.05, ease: [0.25, 1, 0.5, 1] }}
                      className="rounded-2xl p-4 bg-white border border-brand/10 hover:border-primary/20 transition-colors duration-200 space-y-2.5"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1.5 flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <h5 className="text-sm font-bold text-foreground font-sans">{facility.name}</h5>
                            <span className={cn("text-xs font-bold px-2 py-0.5 rounded-full border", config.bg, config.text, config.border)}>
                              {config.label}
                            </span>
                            {facility.tier && (
                              <span className="text-xs font-semibold text-muted-foreground bg-muted px-2 py-0.5 rounded-full capitalize">
                                {facility.tier} care
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground font-sans flex items-center gap-1">
                            <MapPin className="h-3 w-3 shrink-0" />
                            <span className="truncate">{facility.address}</span>
                          </p>
                          {facility.relevance_reason && (
                            <p className="text-xs text-primary font-semibold font-sans">{facility.relevance_reason}</p>
                          )}
                        </div>
                        {facility.distance_km != null && (
                          <div className="text-right shrink-0">
                            <span className="text-sm font-bold text-foreground font-mono tabular-nums">{facility.distance_km}</span>
                            <span className="text-xs text-muted-foreground font-sans"> km</span>
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-3 flex-wrap pt-1 border-t border-brand/5">
                        {facility.phone && (
                          <a href={`tel:${facility.phone}`} className={linkClasses}>
                            <Phone className="h-3 w-3" />
                            {facility.phone}
                          </a>
                        )}
                        {facility.website && (
                          <a href={facility.website} target="_blank" rel="noopener noreferrer" className={linkClasses}>
                            <ExternalLink className="h-3 w-3" />
                            Website
                          </a>
                        )}
                        <a
                          href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(facility.name + " " + facility.address)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={cn(linkClasses, "ml-auto")}
                        >
                          <Navigation className="h-3 w-3" />
                          Directions
                        </a>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FacilityRecommendation;
