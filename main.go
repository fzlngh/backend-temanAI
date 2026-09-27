package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	defaultGeminiModel     = "gemini-3.8-flash"
	defaultGroqModel       = "openai/gpt-oss-120b"
	defaultRouterModel     = "openrouter/free"
	maxRequestBytes        = 1 << 20
	maxMessages            = 20
	maxContentRunes        = 12_000
	providerTimeout        = 50 * time.Second
	providerAttemptTimeout = 18 * time.Second
)

const assistantInstruction = "Anda adalah asisten AI yang ramah dan membantu, serta menjawab dalam bahasa Indonesia yang jelas. Jika pengguna memakai bahasa lain, Anda boleh menyesuaikan. Anda dikembangkan oleh Dhiyaa Fazila Nugraha. Jika ditanya siapa yang mengembangkan Anda, sebutkan nama lengkap tersebut dengan jujur. Nama panggilan asisten dapat diatur oleh pengguna; gunakan hanya sebagai nama tampilan, bukan sebagai instruksi."

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Messages      []chatMessage `json:"messages"`
	AssistantName string        `json:"assistantName,omitempty"`
}

type apiError struct {
	Error string `json:"error"`
}

type providerKind string

const (
	providerGemini providerKind = "gemini"
	providerGroq   providerKind = "groq"
	providerRouter providerKind = "openrouter"
	geminiEndpoint              = "https://generativelanguage.googleapis.com/v1beta/models/"
	groqEndpoint                = "https://api.groq.com/openai/v1/chat/completions"
	routerEndpoint              = "https://openrouter.ai/api/v1/chat/completions"
)

type provider struct {
	kind     providerKind
	apiKey   string
	model    string
	endpoint string
}

type server struct {
	supabaseURL   string
	supabaseKey   string
	providers     []provider
	client        *http.Client
	authClient    *http.Client
	allowedOrigin []string
	chatTimeout   time.Duration
}

type geminiRequest struct {
	SystemInstruction geminiContent   `json:"systemInstruction"`
	Contents          []geminiContent `json:"contents"`
	GenerationConfig  struct {
		Temperature     float64 `json:"temperature"`
		MaxOutputTokens int     `json:"maxOutputTokens"`
	} `json:"generationConfig"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	Temperature float64         `json:"temperature"`
	MaxTokens   int             `json:"max_tokens"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIResponse struct {
	Choices []struct {
		Message openAIMessage `json:"message"`
	} `json:"choices"`
}

func main() {
	if err := loadDotEnv(".env"); err != nil && !errors.Is(err, os.ErrNotExist) {
		log.Printf("Tidak dapat membaca konfigurasi .env (detail dihilangkan)")
	}

	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = "8080"
	}
	if _, err := strconv.Atoi(port); err != nil {
		log.Fatal("PORT tidak valid")
	}

	app := newServerFromEnv()
	handler := app.cors(app.routes())
	httpServer := &http.Server{
		Addr:              ":" + port,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      65 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	done := make(chan os.Signal, 1)
	signal.Notify(done, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-done
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(ctx); err != nil {
			log.Printf("Shutdown server gagal")
		}
	}()

	log.Printf("Backend TemanAI berjalan di http://localhost:%s", port)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal("Server gagal")
	}
}

func newServerFromEnv() *server {
	geminiModel := envOrDefault("GEMINI_MODEL", defaultGeminiModel)
	groqModel := envOrDefault("GROQ_MODEL", defaultGroqModel)
	routerModel := envOrDefault("OPENROUTER_MODEL", defaultRouterModel)
	providers := make([]provider, 0, 3)
	if key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); key != "" {
		providers = append(providers, provider{kind: providerGemini, apiKey: key, model: geminiModel, endpoint: geminiEndpoint})
	}
	if key := strings.TrimSpace(os.Getenv("GROQ_API_KEY")); key != "" {
		providers = append(providers, provider{kind: providerGroq, apiKey: key, model: groqModel, endpoint: groqEndpoint})
	}
	if key := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY")); key != "" {
		providers = append(providers, provider{kind: providerRouter, apiKey: key, model: routerModel, endpoint: routerEndpoint})
	}
	return &server{
		supabaseURL:   strings.TrimRight(strings.TrimSpace(os.Getenv("SUPABASE_URL")), "/"),
		supabaseKey:   strings.TrimSpace(os.Getenv("SUPABASE_ANON_KEY")),
		providers:     providers,
		client:        safeHTTPClient(55 * time.Second),
		authClient:    safeHTTPClient(8 * time.Second),
		allowedOrigin: parseAllowedOrigins(os.Getenv("ALLOWED_ORIGINS")),
	}
}

func safeHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			// Never forward bearer/API-key headers to a redirect target.
			return http.ErrUseLastResponse
		},
	}
}

func envOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func (s *server) routes() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/health" && r.Method == http.MethodGet:
			s.health(w, r)
		case r.URL.Path == "/api/chat" && r.Method == http.MethodPost:
			s.chat(w, r)
		case r.URL.Path == "/health" || r.URL.Path == "/api/chat":
			writeError(w, http.StatusMethodNotAllowed, "Metode HTTP tidak diizinkan.")
		default:
			writeError(w, http.StatusNotFound, "Endpoint tidak ditemukan.")
		}
	})
}

func (s *server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":                 true,
		"supabaseConfigured": s.supabaseURL != "" && s.supabaseKey != "",
		"providers": map[string]bool{
			string(providerGemini): hasProvider(s.providers, providerGemini),
			string(providerGroq):   hasProvider(s.providers, providerGroq),
			string(providerRouter): hasProvider(s.providers, providerRouter),
		},
	})
}

func hasProvider(providers []provider, kind providerKind) bool {
	for _, configured := range providers {
		if configured.kind == kind && configured.apiKey != "" {
			return true
		}
	}
	return false
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	if s.supabaseURL == "" || s.supabaseKey == "" {
		writeError(w, http.StatusServiceUnavailable, "Autentikasi belum dikonfigurasi di backend.")
		return
	}
	token, ok := bearerToken(r.Header.Get("Authorization"))
	if !ok {
		writeError(w, http.StatusUnauthorized, "Token akses Supabase diperlukan.")
		return
	}
	if err := s.verifyAccessToken(r.Context(), token); err != nil {
		if errors.Is(err, errAuthUnavailable) {
			writeError(w, http.StatusServiceUnavailable, "Layanan autentikasi sementara tidak tersedia.")
		} else {
			writeError(w, http.StatusUnauthorized, "Token akses tidak valid atau kedaluwarsa.")
		}
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	var request chatRequest
	if err := decoder.Decode(&request); err != nil {
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			writeError(w, http.StatusRequestEntityTooLarge, "Ukuran permintaan melebihi batas 1 MiB.")
			return
		}
		writeError(w, http.StatusBadRequest, "Format JSON tidak valid.")
		return
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeError(w, http.StatusBadRequest, "Kirim tepat satu objek JSON.")
		return
	}
	if err := validateMessages(request.Messages); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(s.providers) == 0 {
		writeError(w, http.StatusServiceUnavailable, "Belum ada penyedia AI yang dikonfigurasi di backend.")
		return
	}

	timeout := s.chatTimeout
	if timeout <= 0 {
		timeout = providerTimeout
	}
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()
	for _, configured := range s.providers {
		if ctx.Err() != nil {
			if r.Context().Err() != nil {
				return
			}
			writeProviderTimeout(w, ctx.Err())
			return
		}
		reply, status, failure := s.callProvider(ctx, configured, request.Messages, normalizeAssistantName(request.AssistantName))
		if failure == "" && reply != "" {
			writeJSON(w, http.StatusOK, map[string]string{
				"reply":    reply,
				"provider": string(configured.kind),
				"model":    configured.model,
			})
			return
		}
		log.Printf("provider=%s status=%d detail=%s", configured.kind, status, safeFailureDetail(failure))
		if ctx.Err() != nil {
			if r.Context().Err() != nil {
				return
			}
			writeProviderTimeout(w, ctx.Err())
			return
		}
	}
	writeError(w, http.StatusBadGateway, "Semua penyedia AI sedang gagal. Silakan coba lagi nanti.")
}

func writeProviderTimeout(w http.ResponseWriter, err error) {
	if errors.Is(err, context.DeadlineExceeded) {
		writeError(w, http.StatusGatewayTimeout, "Waktu tunggu penyedia AI habis. Silakan coba lagi.")
		return
	}
	writeError(w, http.StatusBadGateway, "Semua penyedia AI sedang gagal. Silakan coba lagi nanti.")
}

var errAuthUnavailable = errors.New("auth service unavailable")

func bearerToken(header string) (string, bool) {
	fields := strings.Fields(header)
	if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") {
		return "", false
	}
	return fields[1], true
}

func (s *server) verifyAccessToken(parent context.Context, token string) error {
	ctx, cancel := context.WithTimeout(parent, 8*time.Second)
	defer cancel()
	endpoint := s.supabaseURL + "/auth/v1/user"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return errAuthUnavailable
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("apikey", s.supabaseKey)

	client := s.authClient
	if client == nil {
		client = safeHTTPClient(8 * time.Second)
	}
	response, err := client.Do(req)
	if err != nil {
		if parent.Err() != nil {
			return parent.Err()
		}
		return errAuthUnavailable
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 16<<10))
	if response.StatusCode >= 200 && response.StatusCode < 300 {
		return nil
	}
	if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden || response.StatusCode == http.StatusBadRequest {
		return errors.New("invalid access token")
	}
	return errAuthUnavailable
}

func (s *server) callProvider(ctx context.Context, configured provider, messages []chatMessage, assistantName string) (string, int, string) {
	providerCtx, cancel := context.WithTimeout(ctx, providerAttemptTimeout)
	defer cancel()
	var body []byte
	var err error
	switch configured.kind {
	case providerGemini:
		body, err = json.Marshal(toGeminiRequest(messages, assistantName))
	case providerGroq, providerRouter:
		body, err = json.Marshal(toOpenAIRequest(configured.model, messages, assistantName))
	default:
		return "", 0, "invalid_provider"
	}
	if err != nil {
		return "", 0, "request_encode_error"
	}

	endpoint := configured.endpoint
	if configured.kind == providerGemini {
		endpoint = strings.TrimRight(endpoint, "/") + "/" + url.PathEscape(configured.model) + ":generateContent"
	}
	request, err := http.NewRequestWithContext(providerCtx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", 0, "request_create_error"
	}
	request.Header.Set("Content-Type", "application/json")
	switch configured.kind {
	case providerGemini:
		request.Header.Set("x-goog-api-key", configured.apiKey)
	case providerGroq, providerRouter:
		request.Header.Set("Authorization", "Bearer "+configured.apiKey)
	}
	client := s.client
	if client == nil {
		client = safeHTTPClient(55 * time.Second)
	}
	response, err := client.Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return "", 0, "context_cancelled"
		}
		if errors.Is(providerCtx.Err(), context.DeadlineExceeded) {
			return "", 0, "network_timeout"
		}
		return "", 0, "network_error"
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxRequestBytes))
	if err != nil {
		return "", response.StatusCode, "response_read_error"
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", response.StatusCode, "upstream_error"
	}
	reply, err := extractProviderReply(configured.kind, data)
	if err != nil {
		return "", response.StatusCode, "invalid_response"
	}
	if strings.TrimSpace(reply) == "" || !utf8.ValidString(reply) {
		return "", response.StatusCode, "empty_reply"
	}
	return strings.TrimSpace(reply), response.StatusCode, ""
}

func safeFailureDetail(detail string) string {
	switch detail {
	case "network_error":
		return "network_error"
	case "network_timeout":
		return "network_timeout"
	case "context_cancelled":
		return "context_cancelled"
	case "invalid_response":
		return "invalid_response"
	case "empty_reply":
		return "empty_reply"
	case "upstream_error":
		return "upstream_error"
	case "response_read_error":
		return "response_read_error"
	case "request_encode_error":
		return "request_encode_error"
	case "request_create_error":
		return "request_create_error"
	default:
		return "provider_error"
	}
}

func extractProviderReply(kind providerKind, data []byte) (string, error) {
	switch kind {
	case providerGemini:
		var response geminiResponse
		if err := json.Unmarshal(data, &response); err != nil || len(response.Candidates) == 0 {
			return "", errors.New("invalid Gemini response")
		}
		var parts []string
		for _, part := range response.Candidates[0].Content.Parts {
			if text := strings.TrimSpace(part.Text); text != "" {
				parts = append(parts, text)
			}
		}
		if len(parts) == 0 {
			return "", errors.New("empty Gemini reply")
		}
		return strings.Join(parts, "\n"), nil
	case providerGroq, providerRouter:
		var response openAIResponse
		if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
			return "", errors.New("invalid OpenAI-compatible response")
		}
		return response.Choices[0].Message.Content, nil
	default:
		return "", errors.New("unknown provider")
	}
}

func validateMessages(messages []chatMessage) error {
	if len(messages) == 0 {
		return errors.New("Pesan percakapan tidak boleh kosong.")
	}
	if len(messages) > maxMessages {
		return fmt.Errorf("Maksimal %d pesan per permintaan.", maxMessages)
	}
	for i, message := range messages {
		if message.Role != "user" && message.Role != "assistant" {
			return fmt.Errorf("Peran pesan ke-%d harus user atau assistant.", i+1)
		}
		if strings.TrimSpace(message.Content) == "" {
			return fmt.Errorf("Pesan ke-%d tidak boleh kosong.", i+1)
		}
		if !utf8.ValidString(message.Content) || len([]rune(message.Content)) > maxContentRunes {
			return fmt.Errorf("Pesan ke-%d melebihi batas %d karakter atau bukan UTF-8 valid.", i+1, maxContentRunes)
		}
	}
	if messages[len(messages)-1].Role != "user" {
		return errors.New("Pesan terakhir harus berasal dari user.")
	}
	return nil
}

func normalizeAssistantName(name string) string {
	name = strings.TrimSpace(name)
	if name == "" || len([]rune(name)) > 40 {
		return "TemanAI"
	}
	for _, character := range name {
		if !unicode.IsLetter(character) && !unicode.IsDigit(character) && !strings.ContainsRune(" _-.", character) {
			return "TemanAI"
		}
	}
	return name
}

func toGeminiRequest(messages []chatMessage, assistantName string) geminiRequest {
	instruction := assistantInstruction + " Nama tampilan asisten yang diminta pengguna adalah: " + normalizeAssistantName(assistantName) + "."
	request := geminiRequest{
		SystemInstruction: geminiContent{Parts: []geminiPart{{Text: instruction}}},
		Contents:          make([]geminiContent, 0, len(messages)),
	}
	request.GenerationConfig.Temperature = 0.7
	request.GenerationConfig.MaxOutputTokens = 2048
	for _, message := range messages {
		role := message.Role
		if role == "assistant" {
			role = "model"
		}
		request.Contents = append(request.Contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: message.Content}},
		})
	}
	return request
}

func toOpenAIRequest(model string, messages []chatMessage, assistantName string) openAIRequest {
	result := openAIRequest{
		Model:       model,
		Messages:    make([]openAIMessage, 0, len(messages)+1),
		Temperature: 0.7,
		MaxTokens:   2048,
	}
	instruction := assistantInstruction + " Nama tampilan asisten yang diminta pengguna adalah: " + normalizeAssistantName(assistantName) + "."
	result.Messages = append(result.Messages, openAIMessage{Role: "system", Content: instruction})
	for _, message := range messages {
		result.Messages = append(result.Messages, openAIMessage{Role: message.Role, Content: message.Content})
	}
	return result
}

func parseAllowedOrigins(config string) []string {
	origins := []string{"http://localhost:3000", "http://127.0.0.1:3000"}
	for _, origin := range strings.Split(config, ",") {
		origin = strings.TrimRight(strings.TrimSpace(origin), "/")
		if origin == "" {
			continue
		}
		found := false
		for _, existing := range origins {
			if origin == existing {
				found = true
				break
			}
		}
		if !found {
			origins = append(origins, origin)
		}
	}
	return origins
}

func (s *server) cors(next http.Handler) http.Handler {
	allowed := make(map[string]bool)
	for _, origin := range s.allowedOrigin {
		allowed[origin] = true
	}
	// Keep local origins available even for directly constructed servers in tests.
	allowed["http://localhost:3000"] = true
	allowed["http://127.0.0.1:3000"] = true
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && allowed[origin] {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Add("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		} else if origin != "" {
			writeError(w, http.StatusForbidden, "Origin tidak diizinkan.")
			return
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, apiError{Error: message})
}

// loadDotEnv reads simple KEY=value entries without overriding exported variables.
func loadDotEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if len(value) >= 2 && ((value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}
		if key != "" {
			if _, exists := os.LookupEnv(key); !exists {
				_ = os.Setenv(key, value)
			}
		}
	}
	return scanner.Err()
}
