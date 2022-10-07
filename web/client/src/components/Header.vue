<template>
  <div class="text-center">
    <h1>
      Make recommendations <br />
      by entering your <span class="text-primary">MyAnimeList account</span>!
    </h1>
  </div>
</template>

<script>
export default {
  name: "Header",
  data() {
    return {
      title: "Anime Recommender",
    };
  },
};
</script>

<style scoped>
h1 {
  margin-bottom: 3rem;
}
</style>
