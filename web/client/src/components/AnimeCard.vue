<template>
  <div class="hover-overlay">
  </div>

  <div class="hover-overlay">
    <!-- <img -->
    <!--   src="https://mdbcdn.b-cdn.net/img/new/standard/city/053.webp" -->
    <!--   class="w-100" -->
    <!-- /> -->
    <div class="card">
      <img :src="imgSrc" class="card-img-top" />
      <div class="card-body">
        <h5 class="card-title">{{ title }}</h5>
        <p class="card-text">Predicted Score: {{ predScore }}</p>
        <!-- <a href="#!" class="btn btn-primary">Button</a> -->
      </div>
    <div class="mask"></div>
    </div>
  </div>
</template>
<script>
export default {
  props: {
    imgSrc: String,
    title: String,
    predScore: Number,
  },
  name: "AnimeCard",
};
</script>
<style scoped>
.card {
  width: 18rem;
}
.mask {
  border-radius: 5px;
  background: linear-gradient(
    45deg,
    rgba(29, 236, 197, 0.5),
    rgba(91, 14, 214, 0.5) 100%
  );
}
</style>
