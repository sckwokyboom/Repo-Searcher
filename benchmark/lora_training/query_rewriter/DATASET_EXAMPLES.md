# Dataset Examples Report (v2 - Improved Generator)

**Total samples:** 1490
**Repository:** jdereg/java-util
**Styles:** ['behavioral', 'navigation', 'type_aware', 'short']

## Style Distribution

- **behavioral:** 257 samples (87 test)
- **navigation:** 386 samples (113 test)
- **short:** 347 samples (81 test)
- **type_aware:** 500 samples (125 test)

## Style: behavioral

### Example 1

**Class:** `GraphComparator`
**Method:** `compareMaps`
**File:** `src/main/java/com/cedarsoftware/util/GraphComparator.java`
**Is test:** False
**Semantic tags:** ['conversion', 'security', 'comparison']

**Javadoc:**
```java
/**
     * Deeply compare two Maps and generate the appropriate 'put' or 'remove' commands
     * to rectify their differences.  Order of Maps des not matter from an equality standpoint.
     * So for example, a TreeMap and a HashMap are considered equal (no Deltas) if they contain
     * the same entries, regardless of order.
     */
```

**Method body:**
```java
private static void compareMaps(Delta delta, Collection<Delta> deltas, LinkedList<Delta> stack, ID idFetcher, Map<Object, Object> idCache)
    {
        Map<Object, Object> srcMap = (Map<Object, Object>) delta.srcValue;
        Map<Object, Object> targetMap = (Map<Object, Object>) delta.targetValue;

        // Walk source Map keys and see if they exist in target map.  If not, that entry needs to be removed.
        // If the key exists in both, then the value must tested for equivalence.  If !equal, then a PUT command
        // is created to re-associate target value to key.
        final String sysId = "(" + System.identityHashCode(srcMap) + ')';
        for (Map.Entry<Object, Object> entry : srcMap.entrySet())
        {
            Object srcKey = entry.getKey();
            Object srcValue = entry.getValue();
            String srcPtr = sysId + "['" + System.identityHashCode(srcKey) + "']";

            Object targetValue = targetMap.get(srcKey);
            if (targetValue != nul
// ... (truncated)
```

**Javadoc summary:** deeply compare two Maps and generate the appropriate 'put' or 'remove' commands

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: find code that compares maps

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "find",
    "code",
    "compares",
    "maps"
  ],
  "project_terms": [
    "GraphComparator",
    "Delta",
    "ID",
    "Entry"
  ],
  "method_hints": [
    "compareMaps",
    "identityHashCode",
    "entrySet"
  ],
  "api_hints": [
    "Collection",
    "LinkedList",
    "Map"
  ],
  "search_queries": [
    "find code compares maps",
    "GraphComparator compareMaps",
    "compareMaps Collection LinkedList",
    "conversion compare maps"
  ]
}
```

---

### Example 2

**Class:** `ConcurrentNavigableSetNullSafe`
**Method:** `ConcurrentNavigableSetNullSafe`
**File:** `src/main/java/com/cedarsoftware/util/ConcurrentNavigableSetNullSafe.java`
**Is test:** False
**Semantic tags:** ['concurrency', 'comparison']

**Javadoc:**
```java
/**
     * Constructs a new, empty ConcurrentNavigableSetNullSafe with the specified comparator.
     *
     * @param comparator the comparator that will be used to order this set. If null, the natural
     *                   ordering of the elements will be used.
     */
```

**Method body:**
```java
public ConcurrentNavigableSetNullSafe(Comparator<? super E> comparator) {
        this.originalComparator = comparator;
        this.viewComparator = comparator;
        Comparator<Object> comp = wrapComparator(comparator);
        this.internalSet = new ConcurrentSkipListSet<>(comp);
    }
```

**Javadoc summary:** constructs a new, empty ConcurrentNavigableSetNullSafe with the specified comparator

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: how does the code concurrent navigable set null

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "code",
    "concurrent",
    "navigable",
    "set",
    "null"
  ],
  "project_terms": [
    "ConcurrentNavigableSetNullSafe",
    "ConcurrentSkipListSet"
  ],
  "method_hints": [
    "ConcurrentNavigableSetNullSafe",
    "wrapComparator"
  ],
  "api_hints": [
    "Comparator"
  ],
  "search_queries": [
    "code concurrent navigable set null",
    "ConcurrentNavigableSetNullSafe ConcurrentNavigableSetNullSafe",
    "ConcurrentNavigableSetNullSafe Comparator",
    "concurrency concurrent navigable"
  ]
}
```

---

### Example 3

**Class:** `MultiKeyMapMixedListSetTest`
**Method:** `testReplaceWithMixedListSet`
**File:** `src/test/java/com/cedarsoftware/util/MultiKeyMapMixedListSetTest.java`
**Is test:** True
**Semantic tags:** ['conversion', 'security']

**Javadoc:**
```java
/**
     * Test: Replace operations with mixed List/Set keys
     */
```

**Method body:**
```java
@Test
    void testReplaceWithMixedListSet() {
        MultiKeyMap<String> map = new MultiKeyMap<>();

        map.put(new Object[]{
            Arrays.asList("a", "b"),
            new HashSet<>(Arrays.asList("x", "y"))
        }, "old");

        // Replace with Set in different order should work
        assertEquals("old", map.replace(new Object[]{
            Arrays.asList("a", "b"),
            new HashSet<>(Arrays.asList("y", "x"))
        }, "new"));

        assertEquals("new", map.get(new Object[]{
            Arrays.asList("a", "b"),
            new HashSet<>(Arrays.asList("x", "y"))
        }));

        assertEquals(1, map.size());
    }
```

**Javadoc summary:** test: Replace operations with mixed List/Set keys

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: which code handles test replace with mixed

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "tests",
  "keywords": [
    "code",
    "handles",
    "test",
    "replace",
    "mixed"
  ],
  "project_terms": [
    "MultiKeyMapMixedListSetTest",
    "MultiKeyMap"
  ],
  "method_hints": [
    "testReplaceWithMixedListSet",
    "asList",
    "replace"
  ],
  "api_hints": [
    "HashSet"
  ],
  "search_queries": [
    "code handles test replace mixed",
    "MultiKeyMapMixedListSetTest testReplaceWithMixedListSet",
    "testReplaceWithMixedListSet HashSet",
    "conversion test replace"
  ]
}
```

---

### Example 4

**Class:** `EncryptionUtilities`
**Method:** `fastSHA1`
**File:** `src/main/java/com/cedarsoftware/util/EncryptionUtilities.java`
**Is test:** False
**Semantic tags:** ['validation', 'security', 'collection_ops']

**Javadoc:**
```java
/**
     * Calculates a SHA-1 hash of a file using optimized I/O operations.
     * <p>
     * This implementation uses:
     * <ul>
     *   <li>Heap ByteBuffer for efficient memory use</li>
     *   <li>FileChannel for optimal file access</li>
     *   <li>Fallback for non-standard filesystems</li>
     * </ul>
     *
     * @param file the file to hash
     * @return hexadecimal string of the SHA-1 hash, or null if the file cannot be read
     */
```

**Method body:**
```java
public static String fastSHA1(File file) {
        // Security: Validate file size to prevent resource exhaustion
        validateFileSize(file);

        try (FileInputStream fis = new FileInputStream(file)) {
            return calculateFileHash(fis.getChannel(), getSHA1Digest());
        } catch (FileNotFoundException e) {
            return null;
        } catch (IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }
```

**Javadoc summary:** calculates a SHA-1 hash of a file using optimized I/O operations

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: which code handles fast sha1

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "code",
    "handles",
    "fast",
    "sha1"
  ],
  "project_terms": [
    "EncryptionUtilities",
    "FileInputStream",
    "FileNotFoundException",
    "java"
  ],
  "method_hints": [
    "fastSHA1",
    "validateFileSize",
    "calculateFileHash"
  ],
  "api_hints": [
    "File",
    "IOException"
  ],
  "search_queries": [
    "code handles fast sha1",
    "EncryptionUtilities fastSHA1",
    "fastSHA1 File IOException",
    "validation fast sha1"
  ]
}
```

---

### Example 5

**Class:** `SystemUtilities`
**Method:** `getEnvironmentVariables`
**File:** `src/main/java/com/cedarsoftware/util/SystemUtilities.java`
**Is test:** False
**Semantic tags:** ['conversion', 'security']

**Javadoc:**
```java
/**
     * Get all environment variables with optional filtering and security protection.
     *
     * <p><strong>Security Note:</strong> This method automatically filters out sensitive
     * variables such as passwords, tokens, and credentials to prevent information disclosure.
     * Use {@link #getEnvironmentVariablesUnsafe(Predicate)} if you need access to sensitive
     * variables and have verified the security requirements.</p>
     *
     * @param filter optional predicate to further f
```

**Method body:**
```java
public static Map<String, String> getEnvironmentVariables(Predicate<String> filter) {
        boolean securityFiltering = isSecurityEnabled() && isEnvironmentVariableValidationEnabled();
        Map<String, String> env = System.getenv();
        Map<String, String> result = new LinkedHashMap<>(calculateMapCapacity(env.size()));

        for (Map.Entry<String, String> entry : env.entrySet()) {
            String key = entry.getKey();
            // Security: Filter sensitive variables
            if (securityFiltering && isSensitiveVariable(key)) {
                continue;
            }
            // Apply user filter
            if (filter != null && !filter.test(key)) {
                continue;
            }
            result.put(key, entry.getValue());
        }
        return result;
    }
```

**Javadoc summary:** all environment variables with optional filtering and security protection

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: find code that gets environment variables

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "find",
    "code",
    "gets",
    "environment",
    "variables"
  ],
  "project_terms": [
    "SystemUtilities",
    "Predicate",
    "Entry"
  ],
  "method_hints": [
    "getEnvironmentVariables",
    "isSecurityEnabled",
    "isEnvironmentVariableValidationEnabled"
  ],
  "api_hints": [
    "Map",
    "LinkedHashMap"
  ],
  "search_queries": [
    "find code gets environment variables",
    "SystemUtilities getEnvironmentVariables",
    "getEnvironmentVariables Map LinkedHashMap",
    "conversion get environment"
  ]
}
```

---

## Style: navigation

### Example 1

**Class:** `ClassUtilities`
**Method:** `findConvertibleMatches`
**File:** `src/main/java/com/cedarsoftware/util/ClassUtilities.java`
**Is test:** False
**Semantic tags:** ['conversion']

**Javadoc:**
```java
/**
     * Find matches that require type conversion
     */
```

**Method body:**
```java
private static void findConvertibleMatches(Converter converter, Object[] values, boolean[] valueUsed,
                                               Parameter[] parameters, int paramOffset, int paramCount,
                                               boolean[] parameterMatched, Object[] result,
                                               int valueStartInclusive, int valueEndExclusive) {
        for (int i = 0; i < paramCount; i++) {
            if (parameterMatched[i]) continue;

            Class<?> paramType = parameters[paramOffset + i].getType();

            for (int j = valueStartInclusive; j < valueEndExclusive; j++) {
                if (valueUsed[j]) continue;

                Object value = values[j];
                if (value == null) continue;

                Class<?> valueClass = value.getClass();

                if (converter.isSimpleTypeConversionSupported(paramType, valueClass)) {
                    try {
                        Object converted = converter.conv
// ... (truncated)
```

**Javadoc summary:** matches that require type conversion

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: matches that require type conversion

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "matches",
    "require",
    "type",
    "conversion"
  ],
  "project_terms": [
    "ClassUtilities",
    "Converter",
    "Parameter"
  ],
  "method_hints": [
    "findConvertibleMatches",
    "getType",
    "isSimpleTypeConversionSupported"
  ],
  "api_hints": [
    "Exception"
  ],
  "search_queries": [
    "matches require type conversion",
    "ClassUtilities findConvertibleMatches",
    "findConvertibleMatches Exception",
    "conversion find convertible"
  ]
}
```

---

### Example 2

**Class:** `UniqueIdGenerator`
**Method:** `nextId`
**File:** `src/main/java/com/cedarsoftware/util/UniqueIdGenerator.java`
**Is test:** False
**Semantic tags:** ['concurrency', 'date_time', 'comparison']

**Javadoc:**
```java
/**
     * Lock-free generator that preserves the decimal structure and the serverId suffix.
     * It also never advances the timestamp into the future: if a millisecond's sequence is exhausted, we wait.
     */
```

**Method body:**
```java
private static long nextId(AtomicLong lastId, long factor, int perMsLimit, long maxMillis) {
        long now = currentTimeMillis();
        for (;;) {
            final long prev = lastId.get();

            // Compute the millisecond to use: never go backwards relative to last issued ID
            final long prevMs = prev / factor;
            final long baseMs = Math.max(now, prevMs);
            if (baseMs > maxMillis) {
                throw new IllegalStateException("UniqueId range exhausted for factor=" + factor + " on this JVM");
            }

            final long base = baseMs * factor + serverId;

            // If previous ID was in this same ms, bump sequence by +1 step; else start at sequence=0
            long cand = base;
            if (prev >= base) {
                long seqIndex = ((prev - base) / SEQUENCE_STEP) + 1; // next sequence slot
                cand = base + (seqIndex * SEQUENCE_STEP);

                // Sequence capacity exhausted for this millisecond
// ... (truncated)
```

**Javadoc summary:** lock-free generator that preserves the decimal structure and the serverId suffix

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: lock-free generator that preserves the decimal structure and

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "lock-free",
    "generator",
    "preserves",
    "decimal",
    "structure"
  ],
  "project_terms": [
    "UniqueIdGenerator",
    "IllegalStateException"
  ],
  "method_hints": [
    "nextId",
    "currentTimeMillis",
    "max"
  ],
  "api_hints": [
    "AtomicLong"
  ],
  "search_queries": [
    "lock-free generator preserves decimal structure",
    "UniqueIdGenerator nextId",
    "nextId AtomicLong",
    "concurrency next id"
  ]
}
```

---

### Example 3

**Class:** `EncryptionUtilities`
**Method:** `encryptBytes`
**File:** `src/main/java/com/cedarsoftware/util/EncryptionUtilities.java`
**Is test:** False
**Semantic tags:** ['validation', 'security']

**Javadoc:**
```java
/**
     * Encrypts a byte array using AES-128.
     *
     * @param key     encryption key
     * @param content bytes to encrypt
     * @return hexadecimal string of encrypted data
     * @throws IllegalStateException if encryption fails
     */
```

**Method body:**
```java
public static String encryptBytes(String key, byte[] content) {
        if (key == null || content == null) {
            throw new IllegalArgumentException("key and content cannot be null");
        }
        try {
            // Security: Use configurable salt and IV sizes with validation
            int saltSize = STANDARD_SALT_SIZE;
            int ivSize = STANDARD_IV_SIZE;
            validateCryptoParameterSize(saltSize, "Salt", getMinSaltSize(), getMaxSaltSize());
            validateCryptoParameterSize(ivSize, "IV", getMinIvSize(), getMaxIvSize());

            byte[] salt = new byte[saltSize];
            SECURE_RANDOM.nextBytes(salt);
            byte[] iv = new byte[ivSize];
            SECURE_RANDOM.nextBytes(iv);

            SecretKeySpec sKey = new SecretKeySpec(deriveKey(key, salt, 128), "AES");
            Cipher cipher = Cipher.getInstance(AES_GCM_ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, sKey, new GCMParameterSpec(GCM_TAG_BIT_LENGTH, iv));
           
// ... (truncated)
```

**Javadoc summary:** encrypts a byte array using AES-128

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: encrypts a byte array using AES-128

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "encrypts",
    "byte",
    "array",
    "using",
    "aes-128"
  ],
  "project_terms": [
    "EncryptionUtilities",
    "SecretKeySpec",
    "Cipher",
    "GCMParameterSpec"
  ],
  "method_hints": [
    "encryptBytes",
    "validateCryptoParameterSize",
    "getMinSaltSize"
  ],
  "api_hints": [
    "IllegalArgumentException",
    "Exception"
  ],
  "search_queries": [
    "encrypts byte array using aes-128",
    "EncryptionUtilities encryptBytes",
    "encryptBytes IllegalArgumentException Exception",
    "validation encrypt bytes"
  ]
}
```

---

### Example 4

**Class:** `CollectionConversions`
**Method:** `arrayToCollection`
**File:** `src/main/java/com/cedarsoftware/util/convert/CollectionConversions.java`
**Is test:** False
**Semantic tags:** ['conversion', 'security', 'concurrency']

**Javadoc:**
```java
/**
     * Converts an array to a collection, supporting special collection types
     * and nested arrays. Uses iterative processing to handle deeply nested
     * structures without stack overflow. Preserves circular references.
     *
     * @param array      The source array to convert
     * @param targetType The target collection type
     * @param <T>        The collection class to return
     * @return A collection of the specified target type
     */
```

**Method body:**
```java
@SuppressWarnings("unchecked")
    public static <T extends Collection<?>> T arrayToCollection(Object array, Class<T> targetType) {
        // Track visited arrays to handle circular references
        IdentityHashMap<Object, Object> visited = new IdentityHashMap<>();

        // Determine if the target type requires unmodifiable behavior
        boolean requiresUnmodifiable = isUnmodifiable(targetType);
        boolean requiresSynchronized = isSynchronized(targetType);

        // Create the appropriate collection using CollectionHandling
        Collection<Object> rootCollection = (Collection<Object>) createCollection(array, targetType);

        // If the target represents an empty collection, return it immediately
        if (isEmptyCollection(targetType)) {
            return (T) rootCollection;
        }

        // Track source array → target collection mapping
        visited.put(array, rootCollection);

        // Work queue for iterative processing
        Deque<ArrayToCollec
// ... (truncated)
```

**Javadoc summary:** an array to a collection, supporting special collection types

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: an array to a collection, supporting special collection

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "array",
    "collection",
    "supporting",
    "special"
  ],
  "project_terms": [
    "CollectionConversions",
    "IdentityHashMap",
    "Deque",
    "ArrayToCollectionWorkItem"
  ],
  "method_hints": [
    "arrayToCollection",
    "isUnmodifiable",
    "isSynchronized"
  ],
  "api_hints": [
    "Collection"
  ],
  "search_queries": [
    "array collection supporting special",
    "CollectionConversions arrayToCollection",
    "arrayToCollection Collection",
    "conversion array to"
  ]
}
```

---

### Example 5

**Class:** `TestVisitedSetConcurrency`
**Method:** `testMultiThreadedComparisonsNoConcurrentMod`
**File:** `src/test/java/com/cedarsoftware/util/TestVisitedSetConcurrency.java`
**Is test:** True
**Semantic tags:** ['security', 'concurrency', 'comparison']

**Javadoc:**
```java
/**
     * Test multi-threaded comparison without concurrent modification.
     * This verifies that ConcurrentSet doesn't break normal multi-threaded usage.
     */
```

**Method body:**
```java
@Test
    public void testMultiThreadedComparisonsNoConcurrentMod() throws Exception {
        final int THREAD_COUNT = 10;
        final int ITERATIONS = 100;
        final AtomicInteger successCount = new AtomicInteger(0);
        final CountDownLatch startLatch = new CountDownLatch(1);
        final CountDownLatch doneLatch = new CountDownLatch(THREAD_COUNT);

        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);

        for (int i = 0; i < THREAD_COUNT; i++) {
            final int threadId = i;
            executor.submit(() -> {
                try {
                    startLatch.await();

                    for (int j = 0; j < ITERATIONS; j++) {
                        Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));
                        Set<Integer> set2 = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5));

                        if (DeepEquals.deepEquals(set1, set2)) {
                            successCount.incrementAndGet();
       
// ... (truncated)
```

**Javadoc summary:** test multi-threaded comparison without concurrent modification

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: test multi-threaded comparison without concurrent modification

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "tests",
  "keywords": [
    "test",
    "multi-threaded",
    "comparison",
    "without",
    "concurrent",
    "modification"
  ],
  "project_terms": [
    "TestVisitedSetConcurrency"
  ],
  "method_hints": [
    "testMultiThreadedComparisonsNoConcurrentMod",
    "newFixedThreadPool",
    "submit"
  ],
  "api_hints": [
    "Exception",
    "AtomicInteger",
    "CountDownLatch",
    "ExecutorService"
  ],
  "search_queries": [
    "test multi-threaded comparison without concurrent",
    "TestVisitedSetConcurrency testMultiThreadedComparisonsNoConcurrentMod",
    "testMultiThreadedComparisonsNoConcurrentMod Exception AtomicInteger",
    "security test multi"
  ]
}
```

---

## Style: short

### Example 1

**Class:** `CompactMapEqualsSymmetryTest`
**Method:** `testEqualsSymmetrySameCaseKeys`
**File:** `src/test/java/com/cedarsoftware/util/CompactMapEqualsSymmetryTest.java`
**Is test:** True
**Semantic tags:** ['conversion', 'security']

**Javadoc:**
```java
/**
     * When keys match exactly (same case), both directions should return true.
     */
```

**Method body:**
```java
@Test
    void testEqualsSymmetrySameCaseKeys() {
        CompactMap<String, Integer> compact = CompactMap.<String, Integer>builder()
                .caseSensitive(false)
                .build();
        compact.put("id", 1);

        Map<String, Integer> hash = new HashMap<>();
        hash.put("id", 1);

        assertTrue(compact.equals(hash), "Same-case keys should be equal");
        assertTrue(hash.equals(compact), "Same-case keys should be equal (reverse)");
    }
```

**Javadoc summary:** when keys match exactly (same case), both directions should return true

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: test equals symmetry same

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "tests",
  "keywords": [
    "test",
    "equals",
    "symmetry",
    "same"
  ],
  "project_terms": [
    "CompactMapEqualsSymmetryTest",
    "CompactMap"
  ],
  "method_hints": [
    "testEqualsSymmetrySameCaseKeys",
    "build",
    "caseSensitive"
  ],
  "api_hints": [
    "Map",
    "HashMap"
  ],
  "search_queries": [
    "test equals symmetry same",
    "CompactMapEqualsSymmetryTest testEqualsSymmetrySameCaseKeys",
    "testEqualsSymmetrySameCaseKeys Map HashMap",
    "conversion test equals"
  ]
}
```

---

### Example 2

**Class:** `UrlUtilities`
**Method:** `getContentFromUrl`
**File:** `src/main/java/com/cedarsoftware/util/UrlUtilities.java`
**Is test:** False
**Semantic tags:** ['conversion', 'validation', 'parsing']

**Javadoc:**
```java
/**
     * Get content from the passed in URL.  This code will open a connection to
     * the passed in server, fetch the requested content, and return it as a
     * byte[].
     *
     * @param url           URL to hit
     * @param inCookies     Map of session cookies (or null if not needed)
     * @param outCookies    Map of session cookies (or null if not needed)
     * @param allowAllCerts override certificate validation?
     * @return byte[] of content fetched from URL.
     */
```

**Method body:**
```java
@SuppressWarnings("unchecked")
    public static byte[] getContentFromUrl(URL url, Map inCookies, Map outCookies, boolean allowAllCerts) {
        URLConnection c = null;
        try {
            c = getConnection(url, inCookies, true, false, false, allowAllCerts);

            FastByteArrayOutputStream out = new FastByteArrayOutputStream(65536);
            InputStream stream = IOUtilities.getInputStream(c);
            
            // Security: Validate Content-Length header after connection is established
            validateContentLength(c);
            
            // Security: Use size-limited transfer to prevent memory exhaustion
            transferWithLimit(stream, out, maxDownloadSize);
            stream.close();

            if (outCookies != null) {   // [optional] Fetch cookies from server and update outCookie Map (pick up JSESSIONID, other headers)
                getCookies(c, outCookies);
            }

            return out.toByteArray();
        } catch (SSLHandsha
// ... (truncated)
```

**Javadoc summary:** content from the passed in URL

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: get content url connection

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "get",
    "content",
    "url",
    "connection"
  ],
  "project_terms": [
    "UrlUtilities",
    "URLConnection",
    "FastByteArrayOutputStream",
    "SSLHandshakeException"
  ],
  "method_hints": [
    "getContentFromUrl",
    "getConnection",
    "getInputStream"
  ],
  "api_hints": [
    "URL",
    "Map",
    "InputStream",
    "Exception"
  ],
  "search_queries": [
    "get content url connection",
    "UrlUtilities getContentFromUrl",
    "getContentFromUrl URL Map",
    "conversion get content"
  ]
}
```

---

### Example 3

**Class:** `SystemUtilities`
**Method:** `checkReflectionPermission`
**File:** `src/main/java/com/cedarsoftware/util/SystemUtilities.java`
**Is test:** False
**Semantic tags:** ['validation', 'security']

**Javadoc:**
```java
/**
     * Checks security manager permissions for reflection operations.
     *
     * @throws SecurityException if reflection is not permitted
     */
```

**Method body:**
```java
private static void checkReflectionPermission() {
        SecurityManager sm = System.getSecurityManager();
        if (sm != null) {
            sm.checkPermission(new RuntimePermission("accessDeclaredMembers"));
        }
    }
```

**Javadoc summary:** security manager permissions for reflection operations

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: check reflection permission

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "check",
    "reflection",
    "permission"
  ],
  "project_terms": [
    "SystemUtilities",
    "RuntimePermission"
  ],
  "method_hints": [
    "checkReflectionPermission",
    "getSecurityManager",
    "checkPermission"
  ],
  "api_hints": [
    "SecurityManager"
  ],
  "search_queries": [
    "check reflection permission",
    "SystemUtilities checkReflectionPermission",
    "checkReflectionPermission SecurityManager",
    "validation check reflection"
  ]
}
```

---

### Example 4

**Class:** `CompactMap`
**Method:** `removeFromMap`
**File:** `src/main/java/com/cedarsoftware/util/CompactMap.java`
**Is test:** False
**Semantic tags:** ['conversion', 'collection_ops']

**Javadoc:**
```java
/**
     * Removes entry from map storage and handles transition to array if needed.
     * <p>
     * If size after removal equals compactSize, transitions back to array storage.
     * Otherwise, maintains map storage with entry removed.
     * </p>
     *
     * @param map the current map storage
     * @param key the key to remove
     * @return the value associated with the removed key, or null if key not found
     */
```

**Method body:**
```java
private V removeFromMap(Map<K, V> map, Object key) {
        V save = map.remove(key);
        if (save == null && !map.containsKey(key)) {
            return null;
        }

        if (map.isEmpty()) {
            val = EMPTY_MAP;
            return save;
        }

        if (map.size() == compactSize()) {   // Transition back to Object[]
            Object[] entries = new Object[compactSize() * 2];
            int idx = 0;
            for (Entry<K, V> entry : map.entrySet()) {
                entries[idx] = entry.getKey();
                entries[idx + 1] = entry.getValue();
                idx += 2;
            }
            sortCompactArray(entries);
            val = entries;
        }
        return save;
    }
```

**Javadoc summary:** removes entry from map storage and handles transition to array if needed

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: remove map contains key

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "remove",
    "map",
    "contains",
    "key"
  ],
  "project_terms": [
    "CompactMap",
    "Entry"
  ],
  "method_hints": [
    "removeFromMap",
    "containsKey",
    "compactSize"
  ],
  "api_hints": [
    "Map"
  ],
  "search_queries": [
    "remove map contains key",
    "CompactMap removeFromMap",
    "removeFromMap Map",
    "conversion remove from"
  ]
}
```

---

### Example 5

**Class:** `FastReader`
**Method:** `readUntil`
**File:** `src/main/java/com/cedarsoftware/util/FastReader.java`
**Is test:** False
**Semantic tags:** ['parsing']

**Javadoc:**
```java
/**
     * Reads characters into the destination array until one of the two delimiter characters is found.
     * The delimiter character is NOT consumed - it remains available for the next read() call.
     * This method is optimized for scanning strings where we want to read until we hit a quote or backslash.
     *
     * @param dest the destination buffer to read characters into
     * @param off the offset in the destination buffer to start writing
     * @param maxLen the maximum number of
```

**Method body:**
```java
public int readUntil(final char[] dest, int off, int maxLen, char delim1, char delim2) {
        if (in == null) {
            ExceptionUtilities.uncheckedThrow(new IOException("in is null"));
        }

        int totalRead = 0;
        final char[] locBuf = buf;

        // First, drain any pushback buffer
        while (pushbackPosition < pushbackBufferSize && totalRead < maxLen) {
            char c = pushbackBuffer[pushbackPosition];
            if (c == delim1 || c == delim2) {
                // Found delimiter in pushback - don't consume it
                return Math.max(totalRead, 0);
            }
            dest[off++] = c;
            pushbackPosition++;
            totalRead++;
        }

        // Now read from main buffer
        while (totalRead < maxLen) {
            fill();
            if (limit == -1) {
                // EOF reached
                return totalRead > 0 ? totalRead : -1;
            }

            // Scan for delimiter in a tight loop (reads onl
// ... (truncated)
```

**Javadoc summary:** reads characters into the destination array until one of the two delimiter characters is found

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: read until unchecked

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "read",
    "until",
    "unchecked"
  ],
  "project_terms": [
    "FastReader"
  ],
  "method_hints": [
    "readUntil",
    "uncheckedThrow",
    "max"
  ],
  "api_hints": [
    "IOException"
  ],
  "search_queries": [
    "read until unchecked",
    "FastReader readUntil",
    "readUntil IOException",
    "parsing read until"
  ]
}
```

---

## Style: type_aware

### Example 1

**Class:** `ConverterEverythingTest`
**Method:** `loadBooleanTests`
**File:** `src/test/java/com/cedarsoftware/util/convert/ConverterEverythingTest.java`
**Is test:** True
**Semantic tags:** ['conversion', 'concurrency']

**Javadoc:**
```java
/**
     * Boolean/boolean
     */
```

**Method body:**
```java
private static void loadBooleanTests() {
        TEST_DB.put(pair(Void.class, boolean.class), new Object[][]{
                {null, false},
        });
        TEST_DB.put(pair(Void.class, Boolean.class), new Object[][]{
                {null, null},
        });
        TEST_DB.put(pair(Byte.class, Boolean.class), new Object[][]{
                {(byte) -2, true},
                {(byte) -1, true},
                {(byte) 0, false, true},
                {(byte) 1, true, true},
                {(byte) 2, true},
        });
        TEST_DB.put(pair(Short.class, Boolean.class), new Object[][]{
                {(short) -2, true},
                {(short) -1, true },
                {(short) 0, false, true},
                {(short) 1, true, true},
                {(short) 2, true},
        });
        TEST_DB.put(pair(Integer.class, Boolean.class), new Object[][]{
                {-2, true},
                {-1, true},
                {0, false, true},
                {1, true, true},
  
// ... (truncated)
```

**Javadoc summary:** boolean/boolean

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: pair map atomic boolean

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "tests",
  "keywords": [
    "pair",
    "map",
    "atomic",
    "boolean"
  ],
  "project_terms": [
    "ConverterEverythingTest",
    "UUID"
  ],
  "method_hints": [
    "loadBooleanTests",
    "pair",
    "mapOf"
  ],
  "api_hints": [
    "AtomicBoolean",
    "AtomicInteger",
    "AtomicLong",
    "BigInteger"
  ],
  "search_queries": [
    "pair map atomic boolean",
    "ConverterEverythingTest loadBooleanTests",
    "loadBooleanTests AtomicBoolean AtomicInteger",
    "conversion load boolean"
  ]
}
```

---

### Example 2

**Class:** `EncryptionUtilities`
**Method:** `encryptBytes`
**File:** `src/main/java/com/cedarsoftware/util/EncryptionUtilities.java`
**Is test:** False
**Semantic tags:** ['validation', 'security']

**Javadoc:**
```java
/**
     * Encrypts a byte array using AES-128.
     *
     * @param key     encryption key
     * @param content bytes to encrypt
     * @return hexadecimal string of encrypted data
     * @throws IllegalStateException if encryption fails
     */
```

**Method body:**
```java
public static String encryptBytes(String key, byte[] content) {
        if (key == null || content == null) {
            throw new IllegalArgumentException("key and content cannot be null");
        }
        try {
            // Security: Use configurable salt and IV sizes with validation
            int saltSize = STANDARD_SALT_SIZE;
            int ivSize = STANDARD_IV_SIZE;
            validateCryptoParameterSize(saltSize, "Salt", getMinSaltSize(), getMaxSaltSize());
            validateCryptoParameterSize(ivSize, "IV", getMinIvSize(), getMaxIvSize());

            byte[] salt = new byte[saltSize];
            SECURE_RANDOM.nextBytes(salt);
            byte[] iv = new byte[ivSize];
            SECURE_RANDOM.nextBytes(iv);

            SecretKeySpec sKey = new SecretKeySpec(deriveKey(key, salt, 128), "AES");
            Cipher cipher = Cipher.getInstance(AES_GCM_ALGORITHM);
            cipher.init(Cipher.ENCRYPT_MODE, sKey, new GCMParameterSpec(GCM_TAG_BIT_LENGTH, iv));
           
// ... (truncated)
```

**Javadoc summary:** encrypts a byte array using AES-128

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: validate crypto parameter size

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "validate",
    "crypto",
    "parameter",
    "size"
  ],
  "project_terms": [
    "EncryptionUtilities",
    "SecretKeySpec",
    "Cipher",
    "GCMParameterSpec"
  ],
  "method_hints": [
    "encryptBytes",
    "validateCryptoParameterSize",
    "getMinSaltSize"
  ],
  "api_hints": [
    "IllegalArgumentException",
    "Exception"
  ],
  "search_queries": [
    "validate crypto parameter size",
    "EncryptionUtilities encryptBytes",
    "encryptBytes IllegalArgumentException Exception",
    "validation encrypt bytes"
  ]
}
```

---

### Example 3

**Class:** `DateUtilities`
**Method:** `parseDate`
**File:** `src/main/java/com/cedarsoftware/util/DateUtilities.java`
**Is test:** False
**Semantic tags:** ['conversion', 'validation', 'parsing']

**Javadoc:**
```java
/**
     * Main API. Retrieve date-time from passed in String.  The boolean ensureDateTimeAlone, if set true, ensures that
     * no other non-date content existed in the String.
     * @param dateStr String containing a date.  See DateUtilities class Javadoc for all the supported formats.
     * @param defaultZoneId ZoneId to use if no timezone offset or name is given.  Cannot be null.
     * @param ensureDateTimeAlone If true, if there is excess non-Date content, it will throw an IllegalArgume
```

**Method body:**
```java
public static ZonedDateTime parseDate(String dateStr, ZoneId defaultZoneId, boolean ensureDateTimeAlone) {
        dateStr = StringUtilities.trimToNull(dateStr);
        if (dateStr == null) {
            return null;
        }
        Convention.throwIfNull(defaultZoneId, "ZoneId cannot be null.  Use ZoneId.of(\"America/New_York\"), ZoneId.systemDefault(), etc.");

        // Security: Input validation to prevent excessively long input strings
        if (isSecurityEnabled() && isInputValidationEnabled()) {
            int maxLength = getMaxInputLength();
            if (dateStr.length() > maxLength) {
                throw new SecurityException("Date string too long (max " + maxLength + " characters): " + dateStr.length());
            }
        }
        
        // Security: Check for malformed input patterns that could cause ReDoS
        if (isSecurityEnabled() && isMalformedStringProtectionEnabled()) {
            validateMalformedInput(dateStr);
        }

        // If purely 
// ... (truncated)
```

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: trim null throw zoned

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "trim",
    "null",
    "throw",
    "zoned"
  ],
  "project_terms": [
    "DateUtilities",
    "ZonedDateTime",
    "ZoneId",
    "SecurityException"
  ],
  "method_hints": [
    "parseDate",
    "trimToNull",
    "throwIfNull"
  ],
  "api_hints": [
    "IllegalArgumentException",
    "Exception"
  ],
  "search_queries": [
    "trim null throw zoned",
    "DateUtilities parseDate",
    "parseDate IllegalArgumentException Exception",
    "conversion parse date"
  ]
}
```

---

### Example 4

**Class:** `IntervalSetExampleTest`
**Method:** `findUnverifiedRanges`
**File:** `src/test/java/com/cedarsoftware/util/IntervalSetExampleTest.java`
**Is test:** True
**Semantic tags:** ['collection_ops', 'date_time']

**Javadoc:**
```java
/**
     * Find gaps in verified ranges within the requested time range
     * This should ONLY work with time ranges, NOT source data
     */
```

**Method body:**
```java
private List<ZonedDateTime[]> findUnverifiedRanges(ZonedDateTime startTime, ZonedDateTime endTime) {
        List<ZonedDateTime[]> unverifiedRanges = new ArrayList<>();
        
        if (verifiedTimeRanges.isEmpty()) {
            // No verified ranges yet - entire range is unverified
            unverifiedRanges.add(new ZonedDateTime[]{startTime, endTime});
            return unverifiedRanges;
        }
        
        // Get all verified intervals sorted by start time
        List<IntervalSet.Interval<ZonedDateTime>> intervals = new ArrayList<>(toList(verifiedTimeRanges));
        intervals.sort(Comparator.comparing(IntervalSet.Interval::getStart));
        
        ZonedDateTime currentPos = startTime;
        
        for (IntervalSet.Interval<ZonedDateTime> interval : intervals) {
            // Skip intervals that end before our range starts
            if (interval.getEnd().isBefore(startTime)) {
                continue;
            }
            
            // Stop if thi
// ... (truncated)
```

**Javadoc summary:** gaps in verified ranges within the requested time range

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: list sort zoned date

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "tests",
  "keywords": [
    "list",
    "sort",
    "zoned",
    "date"
  ],
  "project_terms": [
    "IntervalSetExampleTest",
    "ZonedDateTime",
    "IntervalSet",
    "Interval"
  ],
  "method_hints": [
    "findUnverifiedRanges",
    "toList",
    "sort"
  ],
  "api_hints": [
    "List",
    "ArrayList"
  ],
  "search_queries": [
    "list sort zoned date",
    "IntervalSetExampleTest findUnverifiedRanges",
    "findUnverifiedRanges List ArrayList",
    "collection ops find unverified"
  ]
}
```

---

### Example 5

**Class:** `ReflectionUtils`
**Method:** `call`
**File:** `src/main/java/com/cedarsoftware/util/ReflectionUtils.java`
**Is test:** False
**Semantic tags:** ['reflection', 'security']

**Javadoc:**
```java
/**
     * Provides a simplified, cached reflection API for method invocation using method name.
     * This method combines method lookup and invocation in one step, with results cached
     * for performance.
     * <p>
     * Key features:
     * <ul>
     *     <li>Caches method lookups for improved performance</li>
     *     <li>Handles different classloaders correctly</li>
     *     <li>Converts checked exceptions to runtime exceptions</li>
     *     <li>Caches both successful lookups a
```

**Method body:**
```java
public static Object call(Object instance, String methodName, Object... args) {
        // Security check: Verify permission for reflection access
        Method method = getMethod(instance, methodName, args.length);
        try {
            return method.invoke(instance, args);
        } catch (IllegalAccessException | InvocationTargetException e) {
            ExceptionUtilities.uncheckedThrow(e);
            return null; // never executed
        }
    }
```

**Javadoc summary:** a simplified, cached reflection API for method invocation using method name

**Input (prompt):**
```
Rewrite this search query into structured retrieval hints for a Java codebase.

Query: get method invoke illegal

Output JSON:
```

**Target (completion):**
```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": [
    "get",
    "method",
    "invoke",
    "illegal"
  ],
  "project_terms": [
    "ReflectionUtils",
    "IllegalAccessException",
    "InvocationTargetException"
  ],
  "method_hints": [
    "call",
    "getMethod",
    "invoke"
  ],
  "api_hints": [
    "Method"
  ],
  "search_queries": [
    "get method invoke illegal",
    "ReflectionUtils call",
    "call Method",
    "reflection call"
  ]
}
```

---
