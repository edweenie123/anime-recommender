<script setup>
import { RouterLink, RouterView } from "vue-router";
</script>

<template>
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <!-- Container wrapper -->
    <div class="container">
      <a class="navbar-brand">Anime Recommender</a>
      <!-- <a class="nav-link">About</a> -->

      <ul class="navbar-nav me-auto mb-2 mb-lg-0">
        <li class="nav-item">
          <RouterLink to="/"><a class="nav-link">Home</a></RouterLink>
        </li>
        <li class="nav-item">
          <RouterLink to="/about"><a class="nav-link">About</a></RouterLink>
        </li>
      </ul>
      <div class="d-flex align-items-right">
        <a
          class="btn btn-dark px-3"
          href="https://github.com/edweenie123/anime-recommender"
          role="button"
          target="_blank"
          >GitHub <i class="fab fa-github"></i
        ></a>
      </div>
    </div>
    <!-- Collapsible wrapper -->
    <!-- </div> -->
    <!-- Container wrapper -->
  </nav>
</template>
<script>
export default {
  name: "Navbar",
};
</script>
<style scoped>
.navbar {
  margin-bottom: 5rem;
}
</style>
